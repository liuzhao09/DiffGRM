#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的checkpoint评估脚本
可以加载训练好的模型checkpoint，复用现有的evaluator、beam search等组件进行推理

使用方法:
python eval_checkpoint.py \
    --model=DIFF_GRM \
    --dataset=AmazonReviews2014 \
    --category=Toys_and_Games \
    --checkpoint="saved/AmazonReviews2014_Aug-29-2025_02-52-21/pytorch_model.bin" \
    --beam_search_modes='["confidence"]' \
    --vectorized_beam_search='{"test":{"beam_act":256,"beam_max":256}}'
"""

import argparse
import os
import torch
import json
from datetime import datetime
from torch.utils.data import DataLoader
from accelerate import Accelerator
from logging import getLogger

from genrec.utils import (
    get_config, init_seed, init_logger, init_device,
    get_dataset, get_tokenizer, get_model, get_trainer,
    parse_command_line_args, log
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Eval a saved checkpoint on test split (no training).")
    parser.add_argument('--model', type=str, default='DIFF_GRM', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pytorch_model.bin checkpoint')
    parser.add_argument('--output_file', type=str, default=None, help='Output results to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--prefer_ckpt_arch', action='store_true',
                        help='Try to construct model architecture from checkpoint/config next to it.')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional path to the training config (json/yaml) if not embedded in ckpt.')
    
    # 解析已知参数和未知参数
    args, unparsed = parser.parse_known_args()
    return args, unparsed


def validate_sid_mapping_consistency(tokenizer, dataset):
    """验证SID映射的一致性"""
    print("🔍 Validating SID mapping consistency...")
    
    try:
        # 检查item2tokens映射是否存在
        if not hasattr(tokenizer, 'item2tokens') or not tokenizer.item2tokens:
            print("⚠️  Warning: item2tokens mapping not found")
            return
        
        # 检查tokens2item映射是否存在
        if not hasattr(tokenizer, 'tokens2item') or not tokenizer.tokens2item:
            print("⚠️  Warning: tokens2item mapping not found")
            return
        
        # 检查映射数量是否一致
        n_items = dataset.n_items
        n_mapped_items = len(tokenizer.item2tokens)
        print(f"📊 Items in dataset: {n_items}, Items in mapping: {n_mapped_items}")
        
        if n_mapped_items != n_items - 1:  # 减去PAD token
            print(f"⚠️  Warning: Mapping count mismatch (expected {n_items-1}, got {n_mapped_items})")
        
        # 随机测试几个item的映射一致性
        import random
        test_items = list(tokenizer.item2tokens.keys())[:min(10, len(tokenizer.item2tokens))]
        
        for item in test_items:
            if item in tokenizer.item2tokens:
                tokens = tokenizer.item2tokens[item]
                # 检查反向映射
                if hasattr(tokenizer, 'codebooks_to_item_id'):
                    item_id = tokenizer.codebooks_to_item_id(list(tokens))
                    if item_id is not None:
                        original_item = dataset.id_mapping['id2item'][item_id]
                        if original_item != item:
                            print(f"⚠️  Warning: Mapping inconsistency for item {item}")
                            print(f"    Original: {item}, Mapped back: {original_item}")
                        else:
                            print(f"✅ Item {item} mapping verified")
                    else:
                        print(f"⚠️  Warning: Item {item} tokens are invalid")
        
        print("✅ SID mapping consistency check completed")
        
    except Exception as e:
        print(f"⚠️  Warning: SID mapping consistency check failed: {e}")


def _maybe_fill_arch_from_state_dict(config, sd):
    """
    尽可能从 state_dict 的形状推断关键架构参数：
      - hidden_size / d_model / n_embd
      - n_inner / ffn_hidden
      - encoder_n_layer / decoder_n_layer
      - use_cross_attn
      - n_head（尽力推，非唯一）
    仅在 config 未设置相应字段时填充。
    """
    keys = list(sd.keys())
    if not keys:
        return

    # hidden_size: 通常可以从 self_attn.qkv.weight 形状得到 [3*H, H] 或者 from ln/emb
    def _first_shape_of(prefixes):
        for k in keys:
            for p in prefixes:
                if k.endswith(p) or p in k:
                    t = sd[k]
                    return tuple(t.shape)
        return None

    # 1) hidden_size / d_model / n_embd
    shp = _first_shape_of(["self_attn.qkv.weight", "attn.qkv.weight", "encoder_blocks.0.self_attn.qkv.weight",
                           "decoder_blocks.0.self_attn.qkv.weight", "embedding.weight"])
    if shp is not None and len(shp) == 2:
        # [3*H, H] 或 [vocab_size, H]
        H = shp[1]
        old_val = config.get('n_embd', 'not set')
        config['n_embd'] = H
        config.setdefault('hidden_size', H)
        config.setdefault('d_model', H)
        print(f"  🧩 Inferred n_embd: {old_val} -> {H}")

    # 2) n_inner / ffn_hidden（mlp 的中间维度），典型 [ffn, hidden]
    shp = _first_shape_of(["mlp.c_fc.weight", "feed_forward.c_fc.weight"])
    if shp is not None and len(shp) == 2:
        ffn, hid = shp
        # 当prefer_ckpt_arch时，强制覆盖n_inner，避免被默认值卡住
        if 'n_inner' not in config or config.get('n_inner') != ffn:
            old_val = config.get('n_inner', 'not set')
            config['n_inner'] = ffn
            print(f"  🧩 Inferred n_inner: {old_val} -> {ffn}")
        
        # 同时设置mlp_ratio（如果d_model已知）
        if 'n_embd' in config and config['n_embd'] > 0:
            new_ratio = ffn // config['n_embd']
            if new_ratio * config['n_embd'] == ffn:
                old_ratio = config.get('mlp_ratio', 'not set')
                config['mlp_ratio'] = new_ratio
                print(f"  🧩 Inferred mlp_ratio: {old_ratio} -> {new_ratio}")

    # 3) 层数
    def _max_block_idx(prefix):
        max_idx = -1
        for k in keys:
            if k.startswith(prefix):
                # e.g. encoder_blocks.3.ln_2.weight
                parts = k.split('.')
                if len(parts) > 2 and parts[1].isdigit():
                    max_idx = max(max_idx, int(parts[1]))
        return max_idx + 1 if max_idx >= 0 else None

    enc_layers = _max_block_idx("encoder_blocks.")
    dec_layers = _max_block_idx("decoder_blocks.")
    if enc_layers:
        old_val = config.get('encoder_n_layer', 'not set')
        config['encoder_n_layer'] = enc_layers
        config.setdefault('n_layer_encoder', enc_layers)  # 双写兼容
        print(f"  🧩 Inferred encoder_n_layer: {old_val} -> {enc_layers}")
    if dec_layers:
        old_val = config.get('decoder_n_layer', 'not set')
        config['decoder_n_layer'] = dec_layers
        config.setdefault('n_layer_decoder', dec_layers)  # 双写兼容
        print(f"  🧩 Inferred decoder_n_layer: {old_val} -> {dec_layers}")

    # 4) 是否存在 cross-attn
    use_xattn = any("cross_attn.qkv.weight" in k for k in keys)
    old_val = config.get('use_cross_attn', 'not set')
    config['use_cross_attn'] = bool(use_xattn)
    print(f"  🧩 Inferred use_cross_attn: {old_val} -> {use_xattn}")

    # 5) 估计 n_head（非唯一）：尝试 gcd 拆分
    # qkv.weight: [3*H, H] -> 先拿 H，尝试把 H 拆成 n_head * head_dim
    if 'n_embd' in config:
        H = config['n_embd']
        # 常见 head_dim
        for hd in (128, 96, 64, 48, 32, 16):
            if H % hd == 0:
                old_val = config.get('n_head', 'not set')
                config['n_head'] = H // hd
                print(f"  🧩 Inferred n_head: {old_val} -> {H // hd} (head_dim={hd})")
                break


def validate_checkpoint_path(checkpoint_path):
    """验证checkpoint路径是否存在"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 检查是否是文件
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")
    
    print(f"✅ Found checkpoint: {checkpoint_path}")


def setup_environment(args, cli_config):
    """设置环境和配置"""
    # 强制设置关键配置，确保SID映射一致性
    cli_config.setdefault('force_regenerate_opq', False)  # 关键：不要重新生成量化结果
    
    # 优先从 ckpt/显式传入的 config 恢复训练期架构
    found_cfg = args.config_file
    run_dir = os.path.dirname(args.checkpoint)
    if args.prefer_ckpt_arch and not found_cfg:
        for name in ("config.json", "args.json", "hparams.yaml"):
            p = os.path.join(run_dir, name)
            if os.path.exists(p):
                found_cfg = p
                print(f"🧩 Found run config next to ckpt: {p}")
                break
    
    # 获取配置
    config = get_config(args.model, args.dataset, config_file=found_cfg, config_dict=cli_config)
    
    # 设置设备和加速器
    config['device'], config['use_ddp'] = init_device()
    project_dir = os.path.join(
        config['tensorboard_log_dir'], 
        config["dataset"], 
        f"{config['model']}_EVAL"
    )
    accelerator = Accelerator(log_with='tensorboard', project_dir=project_dir)
    config['accelerator'] = accelerator
    
    # 初始化随机种子和日志
    init_seed(config['rand_seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    
    return config, accelerator, logger


def load_dataset_and_tokenizer(config, args):
    """加载数据集和tokenizer"""
    print(f"📊 Loading dataset: {args.dataset}")
    
    # 加载数据集
    dataset_class = get_dataset(args.dataset)
    dataset = dataset_class(config)
    split_data = dataset.split()
    
    print(f"📊 Dataset loaded: {dataset}")
    print(f"📊 Split sizes: train={len(split_data['train'])}, val={len(split_data['val'])}, test={len(split_data['test'])}")
    
    # 加载tokenizer（关键：会复用训练时的SID映射）
    print(f"🔧 Loading tokenizer: {args.model}")
    tokenizer_class = get_tokenizer(args.model)
    tokenizer = tokenizer_class(config, dataset)
    
    vocab_size = getattr(tokenizer, "vocab_size", None)
    max_len = getattr(tokenizer, "max_token_seq_len", None)
    print(f"🔧 Tokenizer loaded: vocab_size={vocab_size}, max_seq_len={max_len}")
    
    # 验证SID映射一致性
    validate_sid_mapping_consistency(tokenizer, dataset)
    
    return dataset, split_data, tokenizer


def create_dataloaders(split_data, tokenizer, config):
    """创建数据加载器"""
    print("🔄 Creating dataloaders...")
    
    # 只tokenize测试集，节省时间
    test_split = {'test': split_data['test']}
    test_dataset = tokenizer.tokenize(test_split)['test']
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        collate_fn=tokenizer.collate_fn['test']
    )
    
    print(f"🔄 Test dataloader created: {len(test_loader)} batches, batch_size={config['eval_batch_size']}")
    
    return test_loader


def load_model_and_checkpoint(config, dataset, tokenizer, args):
    """加载模型和checkpoint"""
    print(f"🤖 Loading model: {args.model}")
    
    # 先把 ckpt 读出来，看是否带有训练期 config/hparams（一些训练脚本会这么存）
    print(f"💾 Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = None
    
    # 可能的容器形式：{'state_dict':..., 'model':..., 'config':...}
    if isinstance(ckpt, dict):
        if 'config' in ckpt and args.prefer_ckpt_arch:
            print("🧩 Found training config inside checkpoint; merging arch keys.")
            train_cfg = ckpt['config']
            # 只合并"架构相关"的关键字，避免覆盖推理参数
            ARCH_KEYS = {
                'n_embd', 'hidden_size', 'd_model', 'n_inner', 'ffn_hidden', 'mlp_ratio',
                'n_head', 'encoder_n_layer', 'decoder_n_layer', 'n_layer',
                'dropout', 'attn_pdrop', 'resid_pdrop', 'embd_pdrop',
                'norm_type', 'norm_eps', 'act', 'use_cross_attn', 'tie_embeddings'
            }
            for k in ARCH_KEYS:
                if k in train_cfg:
                    config[k] = train_cfg[k]
                    print(f"  🧩 Merged arch key: {k} = {train_cfg[k]}")
        
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:   # 有些人把权重放在 'model'
            state_dict = ckpt['model']
    
    if state_dict is None:
        state_dict = ckpt  # 纯 state_dict
    
    # 如果仍然没有架构文件，又开启 prefer_ckpt_arch，则尝试从权重形状做最小推断（兜底）
    if args.prefer_ckpt_arch:
        try:
            _maybe_fill_arch_from_state_dict(config, state_dict)
        except Exception as e:
            print(f"⚠️  Fallback arch inference failed (safe to ignore if you have a config): {e}")
    
    # 创建模型（此时 config 已尽力对齐训练期架构）
    if args.prefer_ckpt_arch:
        print("🧾 Final architecture config:")
        print(f"  d_model/n_embd: {config.get('d_model') or config.get('n_embd')}")
        print(f"  n_inner: {config.get('n_inner')}")
        print(f"  mlp_ratio: {config.get('mlp_ratio')}")
        print(f"  n_head: {config.get('n_head')}")
        print(f"  encoder_layers: {config.get('encoder_n_layer') or config.get('n_layer_encoder')}")
        print(f"  decoder_layers: {config.get('decoder_n_layer') or config.get('n_layer_decoder') or config.get('n_layer')}")
    
    model_class = get_model(args.model)
    model = model_class(config, dataset, tokenizer)
    
    try:
        n_params = getattr(model, "n_parameters", None)
        if n_params is None:
            n_params = sum(p.numel() for p in model.parameters())
    except Exception:
        n_params = "unknown"
    print(f"🤖 Model created: {n_params} parameters")
    
    # 对 DDP 前缀做兼容
    if not any(k.startswith("module.") for k in model.state_dict().keys()) and \
       any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️  load_state_dict not strict. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        for k in (missing[:10] if len(missing) > 10 else missing):
            print(f"  MISSING: {k}")
        for k in (unexpected[:10] if len(unexpected) > 10 else unexpected):
            print(f"  UNEXPECTED: {k}")
        # 如果你必须严格一致，改回 strict=True；这里只是为了在推断架构时尽量跑起来
    else:
        print("✅ Checkpoint loaded successfully")
    
    return model


def run_evaluation(model, test_loader, config, accelerator, logger, args, tokenizer):
    """运行评估"""
    print("🚀 Starting evaluation...")
    
    # 准备模型和数据加载器
    model, test_loader = accelerator.prepare(model, test_loader)
    
    # 创建trainer（只用于evaluate方法）
    trainer_class = get_trainer(args.model)
    trainer = trainer_class(config, model, tokenizer)
    
    # 获取beam search模式
    modes = config.get("beam_search_modes", ["confidence"])
    print(f"🎯 Beam search modes: {modes}")
    
    # 运行评估
    results = trainer.evaluate(test_loader, split='test')
    
    # 打印结果
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("📈 EVALUATION RESULTS")
        print("="*60)
        
        # 按模式分组显示结果
        for mode in modes:
            mode_suffix = "" if mode == "confidence" else f"_{mode}"
            print(f"\n🎯 Mode: {mode}")
            print("-" * 40)
            
            # 显示主要指标
            for metric in config.get('metrics', ['ndcg', 'recall']):
                for k in config.get('topk', [5, 10]):
                    key = f"{metric}@{k}{mode_suffix}"
                    if key in results:
                        print(f"  {key}: {results[key]:.4f}")
            
            # 显示统计信息
            for stat_key in ['legal_ratio', 'duplicate_ratio', 'dup@10']:
                key = f"{stat_key}{mode_suffix}"
                if key in results:
                    print(f"  {key}: {results[key]:.4f}")
            
            # 显示加权分数（仅confidence模式）
            if mode == "confidence" and "weighted_score" in results:
                print(f"  weighted_score: {results['weighted_score']:.4f}")
        
        print("\n" + "="*60)
    
    return results


def save_results(results, args, config):
    """保存结果到文件"""
    if args.output_file is None:
        return
    
    # 准备保存的数据
    output_data = {
        'checkpoint': args.checkpoint,
        'model': args.model,
        'dataset': args.dataset,
        'config': {
            'eval_batch_size': config['eval_batch_size'],
            'beam_search_modes': config.get('beam_search_modes', ["confidence"]),
            'vectorized_beam_search': config.get('vectorized_beam_search', {}),
            'topk': config.get('topk', [5, 10]),
            'metrics': config.get('metrics', ['ndcg', 'recall'])
        },
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存到文件
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {args.output_file}")


def main():
    """主函数"""
    print("🚀 Starting checkpoint evaluation...")
    
    # 解析参数
    args, unparsed = parse_args()
    cli_config = parse_command_line_args(unparsed)
    
    # 验证checkpoint路径
    validate_checkpoint_path(args.checkpoint)
    
    # 设置环境
    config, accelerator, logger = setup_environment(args, cli_config)
    
    # 记录关键配置
    log(f'[EVAL] Device: {config["device"]}', accelerator, logger)
    log(f'[EVAL] Model: {args.model}, Dataset: {args.dataset}', accelerator, logger)
    log(f'[EVAL] Checkpoint: {args.checkpoint}', accelerator, logger)
    
    # 加载数据集和tokenizer
    dataset, split_data, tokenizer = load_dataset_and_tokenizer(config, args)
    
    # 创建数据加载器
    test_loader = create_dataloaders(split_data, tokenizer, config)
    
    # 加载模型和checkpoint
    model = load_model_and_checkpoint(config, dataset, tokenizer, args)
    
    # 运行评估
    results = run_evaluation(model, test_loader, config, accelerator, logger, args, tokenizer)
    
    # 清理资源
    try:
        trainer = get_trainer(args.model)(config, model, tokenizer)
        if hasattr(trainer, 'end'):
            trainer.end()
    except Exception as e:
        print(f"⚠️  Warning: Failed to cleanup trainer: {e}")
    
    # 保存结果
    save_results(results, args, config)
    
    print("✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main() 