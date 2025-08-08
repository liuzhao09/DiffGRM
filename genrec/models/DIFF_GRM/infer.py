#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DIFF_GRM Offline Inference Script

使用已训练的checkpoint进行推理，支持：
- 自定义beam search参数
- 重复序列检测
- 详细的中间结果输出
- 评估指标计算
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
import json
from datetime import datetime

# 添加项目根目录到path（因为此脚本位于genrec/models/DIFF_GRM/）
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, PROJ_ROOT)

# 导入项目模块
from genrec.utils import get_dataset, get_tokenizer, get_model

# 导入本模块的组件
from genrec.models.DIFF_GRM.model import DIFF_GRM
from genrec.models.DIFF_GRM.tokenizer import DIFF_GRMTokenizer
from genrec.models.DIFF_GRM.evaluator import DIFF_GRMEvaluator
from genrec.models.DIFF_GRM.collate import collate_fn_test
from genrec.models.DIFF_GRM.beam import fast_beam_search_for_eval


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DIFF_GRM Inference")
    
    # 基本参数
    parser.add_argument("--config", default="genrec/models/DIFF_GRM/config_infer.yaml",
                       help="推理配置文件路径")
    parser.add_argument("--checkpoint", required=True,
                       help="模型checkpoint路径 (*.bin)")
    parser.add_argument("--category", default=None,
                       help="数据集category，如果不指定则使用config中的设置")
    
    # 推理参数
    parser.add_argument("--split", default="test", choices=["val", "test"],
                       help="评估数据集split")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="推理batch size，如果不指定则使用config中的eval_batch_size")
    
    # 模型架构参数（需要与训练时一致）
    parser.add_argument("--n_digit", type=int, default=None,
                       help="SID位数，必须与训练时一致")
    parser.add_argument("--encoder_n_layer", type=int, default=None,
                       help="Encoder层数")
    parser.add_argument("--decoder_n_layer", type=int, default=None,
                       help="Decoder层数")
    parser.add_argument("--n_head", type=int, default=None,
                       help="注意力头数")
    parser.add_argument("--n_embd", type=int, default=None,
                       help="嵌入维度")
    parser.add_argument("--n_inner", type=int, default=None,
                       help="前馈网络内部维度")
    parser.add_argument("--codebook_size", type=int, default=None,
                       help="Codebook大小")
    
    # Beam Search参数
    parser.add_argument("--top_k", type=int, default=None,
                       help="最终返回的序列数量，覆盖config中的top_k_final")
    parser.add_argument("--beam_act", type=int, default=None,
                       help="前几步的固定beam数量")
    parser.add_argument("--beam_max", type=int, default=None,
                       help="最大beam数量")
    parser.add_argument("--dedup_strategy", choices=["none", "simple", "weighted"], default="simple",
                       help="去重策略: none(不去重), simple(简单去重), weighted(概率加权去重)")
    
    # 输出控制
    parser.add_argument("--print_duplicates", action="store_true",
                       help="打印重复序列统计")
    parser.add_argument("--print_examples", type=int, default=5,
                       help="打印前N个样本的详细结果")
    parser.add_argument("--save_results", action="store_true",
                       help="保存推理结果到文件")
    parser.add_argument("--output_dir", default=None,
                       help="输出目录，默认与checkpoint同目录")
    
    # 调试参数
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="限制推理样本数量（用于快速测试）")
    
    return parser.parse_args()


def load_config(config_path, args):
    """加载并调整配置"""
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖配置
    if args.category:
        config['category'] = args.category
        print(f"Override category: {args.category}")
    
    if args.batch_size:
        config['eval_batch_size'] = args.batch_size
        print(f"Override batch_size: {args.batch_size}")
    
    # 模型架构参数覆盖（与训练时保持一致）
    if args.n_digit:
        config['n_digit'] = args.n_digit
        print(f"Override n_digit: {args.n_digit}")
    
    if args.encoder_n_layer:
        config['encoder_n_layer'] = args.encoder_n_layer
        print(f"Override encoder_n_layer: {args.encoder_n_layer}")
        
    if args.decoder_n_layer:
        config['decoder_n_layer'] = args.decoder_n_layer
        print(f"Override decoder_n_layer: {args.decoder_n_layer}")
        
    if args.n_head:
        config['n_head'] = args.n_head
        print(f"Override n_head: {args.n_head}")
        
    if args.n_embd:
        config['n_embd'] = args.n_embd
        print(f"Override n_embd: {args.n_embd}")
        
    if args.n_inner:
        config['n_inner'] = args.n_inner
        print(f"Override n_inner: {args.n_inner}")
        
    if args.codebook_size:
        config['codebook_size'] = args.codebook_size
        print(f"Override codebook_size: {args.codebook_size}")
    
    # Beam Search参数覆盖
    if 'vectorized_beam_search' not in config:
        config['vectorized_beam_search'] = {}
    
    if args.top_k:
        config['vectorized_beam_search']['top_k_final'] = args.top_k
        print(f"Override top_k_final: {args.top_k}")
    
    if args.beam_act:
        config['vectorized_beam_search']['beam_act'] = args.beam_act
        print(f"Override beam_act: {args.beam_act}")
        
    if args.beam_max:
        config['vectorized_beam_search']['beam_max'] = args.beam_max
        print(f"Override beam_max: {args.beam_max}")
    
    # 去重策略参数
    config['dedup_strategy'] = args.dedup_strategy
    print(f"Deduplication strategy: {args.dedup_strategy}")
    
    # 确保必要的设备配置
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['num_proc'] = 1  # 推理时使用单进程避免问题
    
    # 确保dataset字段存在（用于get_dataset函数）
    config['dataset'] = 'AmazonReviews2014'
    
    # 与训练时保持一致的初始化（参考pipeline.py）
    from accelerate import Accelerator
    from genrec.utils import init_seed, init_logger
    from logging import getLogger
    
    # 创建accelerator（与训练时一致）
    project_dir = os.path.join(
        config.get('tensorboard_log_dir', 'runs'),
        config.get('dataset', 'AmazonReviews2014'),
        config.get('model', 'DIFF_GRM')
    )
    # 推理时不需要tensorboard日志，简化accelerator创建
    try:
        accelerator = Accelerator(log_with='tensorboard', project_dir=project_dir)
    except:
        # 如果tensorboard不可用，使用无日志的accelerator
        accelerator = Accelerator()
        print("Warning: Tensorboard not available, using accelerator without logging")
    config['accelerator'] = accelerator
    
    # 初始化种子和日志（与训练时一致）
    init_seed(config.get('rand_seed', config.get('seed', 42)), 
              config.get('reproducibility', True))
    init_logger(config)
    logger = getLogger()
    
    print(f"Final config - Category: {config.get('category')}, "
          f"Dataset: {config.get('dataset')}, "
          f"Batch size: {config.get('eval_batch_size')}, "
          f"Device: {config['device']}")
    
    return config


def setup_dataset_and_tokenizer(config):
    """设置数据集和tokenizer"""
    print("Setting up dataset and tokenizer...")
    
    # 获取数据集 - 需要先获取类，然后实例化
    dataset_name = config.get('dataset', 'AmazonReviews2014')  # 默认使用AmazonReviews2014
    dataset_class = get_dataset(dataset_name)
    dataset = dataset_class(config)
    print(f"Dataset: {dataset.__class__.__name__}, Items: {dataset.n_items}")
    
    # 获取tokenizer - 需要先获取类，然后实例化
    model_name = config.get('model', 'DIFF_GRM')
    tokenizer_class = get_tokenizer(model_name)
    tokenizer = tokenizer_class(config, dataset)
    print(f"Tokenizer: {tokenizer.__class__.__name__}, Vocab size: {tokenizer.vocab_size}")
    
    return dataset, tokenizer


def setup_model(config, dataset, tokenizer, checkpoint_path):
    """设置模型并加载checkpoint"""
    print("Setting up model...")
    
    # 创建模型
    model = DIFF_GRM(config, dataset, tokenizer)
    print(f"Model: {model.__class__.__name__}, Parameters: {model.n_parameters}")
    
    # 加载checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理可能的键名差异
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 加载权重
    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully!")
    
    # 移动到设备并设置评估模式
    device = torch.device(config['device'])
    model = model.to(device)
    model.eval()
    
    return model, device


def setup_dataloader(config, dataset, tokenizer, split):
    """设置数据加载器"""
    print(f"Setting up {split} dataloader...")
    
    # 获取tokenized数据集
    datasets = dataset.split()
    tokenized_datasets = tokenizer.tokenize(datasets)
    
    if split not in tokenized_datasets:
        raise ValueError(f"Split '{split}' not found in dataset")
    
    eval_dataset = tokenized_datasets[split]
    print(f"{split.capitalize()} dataset size before filtering: {len(eval_dataset)}")
    
    # 🚀 新增：过滤掉空历史的样本，确保公平评估
    if split in ['val', 'test']:
        def has_valid_history(example):
            history_sid = example['history_sid']
            # 检查是否有非PAD的历史
            if isinstance(history_sid, list):
                # 检查是否所有历史都是PAD (token=0)
                non_pad_count = 0
                for seq in history_sid:
                    if isinstance(seq, list):
                        if any(token != 0 for token in seq):  # 0是PAD token
                            non_pad_count += 1
                return non_pad_count > 0
            return True
        
        # 过滤数据集
        eval_dataset = eval_dataset.filter(has_valid_history)
        print(f"{split.capitalize()} dataset size after filtering: {len(eval_dataset)}")
        print(f"Filtered out samples with empty history for fair evaluation")
    
    # 创建DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        eval_dataset,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        collate_fn=collate_fn_test,
        num_workers=0,  # 推理时使用单进程
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    return dataloader


def analyze_dataset_characteristics(dataloader):
    """分析数据集特性，帮助理解性能差异"""
    print("\n" + "="*50)
    print("DATASET CHARACTERISTICS ANALYSIS")
    print("="*50)
    
    history_lengths = []
    empty_history_count = 0
    
    for batch in dataloader:
        if 'history_sid' in batch:
            history_sid = batch['history_sid']  # [batch_size, max_len, n_digit]
            batch_size = history_sid.size(0)
            
            for i in range(batch_size):
                user_history = history_sid[i]  # [max_len, n_digit]
                # 计算非PAD的历史长度
                non_pad_seqs = 0
                for seq in user_history:
                    if torch.any(seq != 0):  # 0是PAD token
                        non_pad_seqs += 1
                
                history_lengths.append(non_pad_seqs)
                if non_pad_seqs == 0:
                    empty_history_count += 1
    
    if history_lengths:
        import numpy as np
        history_lengths = np.array(history_lengths)
        print(f"Total samples: {len(history_lengths)}")
        print(f"Empty history samples: {empty_history_count} ({empty_history_count/len(history_lengths)*100:.1f}%)")
        print(f"History length stats:")
        print(f"  Mean: {history_lengths.mean():.2f}")
        print(f"  Median: {np.median(history_lengths):.2f}")
        print(f"  Min: {history_lengths.min()}")
        print(f"  Max: {history_lengths.max()}")
        print(f"  Distribution:")
        for length in range(0, min(6, history_lengths.max() + 1)):
            count = np.sum(history_lengths == length)
            print(f"    Length {length}: {count} samples ({count/len(history_lengths)*100:.1f}%)")


def analyze_predictions(predictions, labels, tokenizer, args, config=None):
    """分析预测结果"""
    print("\n" + "="*50)
    print("PREDICTION ANALYSIS")
    print("="*50)
    
    batch_size, top_k, n_digit = predictions.shape
    print(f"Predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 1. 重复序列统计
    if args.print_duplicates:
        print("\n--- DUPLICATE ANALYSIS ---")
        total_duplicates = 0
        
        # 添加详细的beam search分析
        print("\n--- BEAM SEARCH QUALITY ANALYSIS ---")
        
        for i in range(min(3, batch_size)):  # 分析前3个样本
            preds_i = predictions[i]  # [top_k, n_digit]
            label_i = labels[i].tolist()
            
            print(f"\nSample {i} analysis:")
            print(f"  True label: {label_i}")
            
            # 转换为tuple以便去重
            pred_tuples = [tuple(pred.tolist()) for pred in preds_i]
            unique_preds = set(pred_tuples)
            duplicates = len(pred_tuples) - len(unique_preds)
            total_duplicates += duplicates
            
            # 统计每个序列的出现次数和位置
            counter = Counter(pred_tuples) 
            unique_sequences = list(counter.keys())
            
            print(f"  Total predictions: {len(pred_tuples)}")
            print(f"  Unique sequences: {len(unique_sequences)}")
            print(f"  Duplicates: {duplicates}")
            
            # 显示每个唯一序列的详细信息
            print("  Sequence distribution:")
            for j, (seq, count) in enumerate(counter.most_common()):
                is_correct = list(seq) == label_i
                marker = "✓" if is_correct else " "
                first_pos = pred_tuples.index(seq) + 1  # 第一次出现的位置
                print(f"    {j+1:2d}: {list(seq)} (appears {count}x, first@{first_pos}) {marker}")
                if j >= 9:  # 只显示前10个唯一序列
                    break
        
        # 继续统计剩余样本
        for i in range(3, batch_size):
            preds_i = predictions[i]
            pred_tuples = [tuple(pred.tolist()) for pred in preds_i]
            unique_preds = set(pred_tuples)
            duplicates = len(pred_tuples) - len(unique_preds)
            total_duplicates += duplicates
        
        print(f"\nOverall statistics:")
        print(f"Total duplicates across all samples: {total_duplicates}")
        print(f"Average duplicates per sample: {total_duplicates / batch_size:.2f}")
        
        # 根据去重策略给出解释
        dedup_strategy = config.get('dedup_strategy', 'weighted') if config else 'weighted'
        if dedup_strategy == "none":
            print("Note: No deduplication applied - duplicates are expected")
        elif dedup_strategy == "simple":
            print("Note: Simple deduplication - duplicates removed by first occurrence")
        else:  # weighted
            print("Note: Probability-weighted deduplication - duplicate probabilities aggregated")
    
    # 2. 打印详细示例
    if args.print_examples > 0:
        print(f"\n--- DETAILED EXAMPLES (first {args.print_examples}) ---")
        n_examples = min(args.print_examples, batch_size)
        
        for i in range(n_examples):
            print(f"\nSample {i}:")
            print(f"  True label: {labels[i].tolist()}")
            print(f"  Predictions (top-{top_k}):")
            
            for j in range(top_k):
                pred = predictions[i, j].tolist()
                is_correct = pred == labels[i].tolist()
                marker = "✓" if is_correct else " "
                print(f"    {j:2d}: {pred} {marker}")
    
    # 3. 序列分布统计
    print(f"\n--- SEQUENCE STATISTICS ---")
    all_predictions = predictions.view(-1, n_digit)  # [batch_size * top_k, n_digit]
    
    # 统计每个位置的分布
    for digit in range(n_digit):
        digit_values = all_predictions[:, digit]
        unique_values = torch.unique(digit_values)
        print(f"Digit {digit}: {len(unique_values)} unique values "
              f"(range: {digit_values.min().item()}-{digit_values.max().item()})")
    
    return total_duplicates


def run_inference(model, dataloader, device, tokenizer, args):
    """运行推理"""
    print("\n" + "="*50)
    print("RUNNING INFERENCE")
    print("="*50)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    total_batches = len(dataloader)
    if args.max_samples:
        max_batches = (args.max_samples + dataloader.batch_size - 1) // dataloader.batch_size
        total_batches = min(total_batches, max_batches)
        print(f"Limiting to {total_batches} batches ({args.max_samples} samples)")
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Inference", total=total_batches)
        for batch_idx, batch in enumerate(pbar):
            if args.max_samples and batch_idx >= total_batches:
                break
            
            # 移动数据到设备
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # 获取encoder输出
            encoder_outputs = model.forward(batch, return_loss=False)
            encoder_hidden = encoder_outputs.hidden_states
            
            # 🚀 设置当前split为test，使用较大的beam size
            model.config["current_split"] = "test"
            
            # 使用beam search生成序列
            predictions = fast_beam_search_for_eval(
                model=model,
                encoder_hidden=encoder_hidden,
                beam_size=10,  # 这个参数会被config中的top_k_final覆盖
                max_len=model.n_digit,
                tokenizer=tokenizer
            )
            
            # 收集结果
            all_predictions.append(predictions.cpu())
            all_labels.append(batch['labels'].cpu())
            
            # 更新进度条
            pbar.set_postfix({
                'batch': f"{batch_idx+1}/{total_batches}",
                'samples': f"{len(all_predictions) * dataloader.batch_size}"
            })
    
    # 合并所有结果
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"\nInference completed!")
    print(f"Total samples: {len(all_predictions)}")
    print(f"Predictions shape: {all_predictions.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    return all_predictions, all_labels


def evaluate_results(predictions, labels, config, tokenizer):
    """评估结果"""
    print("\n" + "="*50)
    print("EVALUATING RESULTS")
    print("="*50)
    
    # 创建评估器
    evaluator = DIFF_GRMEvaluator(config, tokenizer)
    
    # 计算指标
    results = evaluator.calculate_metrics(predictions, labels)
    
    # 打印结果
    print("Evaluation Results:")
    for metric_name, scores in results.items():
        if metric_name == 'weighted_score':
            avg_score = torch.mean(scores).item()
            print(f"  {metric_name}: {avg_score:.4f}")
        else:
            avg_score = torch.mean(scores).item()
            print(f"  {metric_name}: {avg_score:.4f}")
    
    return results


def save_results(predictions, labels, results, args, config):
    """保存结果"""
    if not args.save_results:
        return
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.checkpoint)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"inference_{args.split}_{timestamp}"
    
    # 保存预测结果
    pred_path = os.path.join(output_dir, f"{base_name}_predictions.pt")
    torch.save({
        'predictions': predictions,
        'labels': labels,
        'config': config,
        'args': vars(args)
    }, pred_path)
    print(f"Predictions saved to: {pred_path}")
    
    # 保存评估结果
    results_path = os.path.join(output_dir, f"{base_name}_results.json")
    results_json = {}
    for key, value in results.items():
        if torch.is_tensor(value):
            results_json[key] = value.mean().item()
        else:
            results_json[key] = value
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results_json,
            'config': config,
            'args': vars(args),
            'timestamp': timestamp
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {results_path}")


def main():
    """主函数"""
    args = get_args()
    
    print("="*60)
    print("DIFF_GRM INFERENCE")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Debug mode: {args.debug}")
    
    try:
        # 1. 加载配置
        config = load_config(args.config, args)
        
        # 2. 设置数据集和tokenizer
        dataset, tokenizer = setup_dataset_and_tokenizer(config)
        
        # 3. 设置模型
        model, device = setup_model(config, dataset, tokenizer, args.checkpoint)
        
        # 4. 设置数据加载器
        dataloader = setup_dataloader(config, dataset, tokenizer, args.split)
        
        # 4.5. 分析数据集特性（调试用）
        if args.debug:
            analyze_dataset_characteristics(dataloader)
        
        # 5. 运行推理
        predictions, labels = run_inference(model, dataloader, device, tokenizer, args)
        
        # 6. 分析预测结果
        total_duplicates = analyze_predictions(predictions, labels, tokenizer, args, config)
        
        # 7. 评估结果
        results = evaluate_results(predictions, labels, config, tokenizer)
        
        # 8. 保存结果
        save_results(predictions, labels, results, args, config)
        
        print("\n" + "="*60)
        print("INFERENCE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred during inference: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 