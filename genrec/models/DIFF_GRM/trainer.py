# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict

from genrec.utils import get_total_steps, get_file_name, config_for_log
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class DIFF_GRMTrainer:
    """
    DIFF_GRM模型的训练器，支持diffusion训练模式
    """

    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # Use the same Accelerator created by Pipeline so logging / is_main_process stay consistent
        self.accelerator = config.get('accelerator', None) or Accelerator()
        
        # 设置保存路径
        self.saved_model_ckpt = os.path.join(
            'saved', get_file_name(config), 'pytorch_model.bin'
        )
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)
        
        # 读取调度配置
        self.schedule_cfg = self.config.get('mask_schedule', {}) or {}
        
        # 处理字符串格式的配置（命令行参数传入的情况）
        if isinstance(self.schedule_cfg, str):
            try:
                import json
                self.schedule_cfg = json.loads(self.schedule_cfg)
            except Exception:
                try:
                    import ast
                    self.schedule_cfg = ast.literal_eval(self.schedule_cfg)
                except Exception:
                    print(f"[WARN] mask_schedule is a string but cannot be parsed: {self.schedule_cfg}. Disable schedule.")
                    self.schedule_cfg = {}
        
        self.schedule_enabled = bool(self.schedule_cfg.get('enabled', False))

    def fit(self, train_dataloader, val_dataloader):
        """
        训练模型 - 适配diffusion模式
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return None, None

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_n_steps,
        )

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        
        self.accelerator.init_trackers(
            project_name=get_file_name(self.config, suffix=''),
            config=config_for_log(self.config),
            init_kwargs={"tensorboard": {"flush_secs": 60}},
        )

        n_epochs = np.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)).astype(int)
        best_epoch = 0
        best_val_score = -1
        
        # ====== 阶段调度器 ======
        def _apply_stage(stage_name: str):
            if stage_name == 'least':
                self.model.set_masking_mode(
                    'guided',
                    guided_select='least',
                    guided_steps=self.schedule_cfg.get('guided_steps', self.config.get('guided_steps', 'auto')),
                    guided_conf_metric=self.schedule_cfg.get('guided_conf_metric', self.config.get('guided_conf_metric', 'msp')),
                    guided_refresh_each_step=self.schedule_cfg.get('guided_refresh_each_step', self.config.get('guided_refresh_each_step', False)),
                )
            elif stage_name == 'sequential':
                self.model.set_masking_mode(
                    'sequential',
                    sequential_steps=self.schedule_cfg.get('seq_steps', self.config.get('sequential_steps', 'auto')),
                    sequential_paths=self.schedule_cfg.get('seq_paths', self.config.get('sequential_paths', 1)),
                )
            elif stage_name == 'most':
                self.model.set_masking_mode(
                    'guided',
                    guided_select='most',
                    guided_steps=self.schedule_cfg.get('guided_steps', self.config.get('guided_steps', 'auto')),
                    guided_conf_metric=self.schedule_cfg.get('guided_conf_metric', self.config.get('guided_conf_metric', 'msp')),
                    guided_refresh_each_step=self.schedule_cfg.get('guided_refresh_each_step', self.config.get('guided_refresh_each_step', False)),
                )
            else:
                raise ValueError(f"Unknown stage: {stage_name}")
            if self.accelerator.is_main_process:
                print(f"[SCHEDULE] >>> Enter stage: {stage_name}")

        # 调度状态
        cur_stage = 'least' if self.schedule_enabled else None
        stage_epoch_run = 0                   # 当前阶段已跑 epoch 数（用于 least 固定5个epoch）
        stage_no_improve_eval = 0             # 当前阶段连续"评估无提升"的计数
        if self.schedule_enabled:
            _apply_stage(cur_stage)
        
        # 新增：跟踪评估次数和无提升的评估次数
        eval_count = 0
        no_improve_count = 0
        
        # 新：若启用调度，用覆盖值；否则用原值
        eval_start_epoch = self.schedule_cfg.get('eval_start_epoch_override', self.config.get('eval_start_epoch', 1)) \
                           if self.schedule_enabled else self.config.get('eval_start_epoch', 1)
        eval_interval = self.schedule_cfg.get('eval_interval_override', self.config['eval_interval']) \
                        if self.schedule_enabled else self.config['eval_interval']

        # 若启用调度，禁用全局 early stopping（由阶段内的 switch_patience_eval 接管）
        use_global_early_stop = (self.config.get('patience', None) is not None) and (not self.schedule_enabled)
        
        self.log(f'[TRAINING] Evaluation config: start from epoch {eval_start_epoch}, interval: {eval_interval}')
        if self.schedule_enabled:
            self.log(f'[TRAINING] Auto schedule enabled: {cur_stage} → sequential → most')

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
            )
            
            for batch in train_progress_bar:
                optimizer.zero_grad()
                
                # Diffusion训练：直接传入包含掩码信息的batch
                outputs = self.model(batch, return_loss=True)
                loss = outputs.loss
                
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.item()

            self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)
            self.log(f'[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_dataloader):.6f}')

            # === Evaluation（保持原评估，但用局部 eval_start_epoch/eval_interval） ===
            if (epoch + 1) >= eval_start_epoch and (epoch + 1) % eval_interval == 0:
                eval_count += 1
                all_results = self.evaluate(val_dataloader, split='val')
                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log({f"Val_Metric/{key}": all_results[key]}, step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')
                    if 'weighted_score' in all_results:
                        ndcg_10 = all_results.get('ndcg@10', 0)
                        recall_10 = all_results.get('recall@10', 0)
                        weighted_score = all_results['weighted_score']
                        self.log(f'[Epoch {epoch + 1}] Weighted Score Details: NDCG@10={ndcg_10:.4f}*0.8 + RECALL@10={recall_10:.4f}*0.2 = {weighted_score:.4f}')
                    self.log(f'[Epoch {epoch + 1}] Evaluation #{eval_count}, Best score: {best_val_score:.4f} (Epoch {best_epoch})')

                # === 保存最优 & 统计是否提升 ===
                val_score = all_results[self.config['val_metric']]
                improved = val_score > best_val_score
                if improved:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    stage_no_improve_eval = 0  # 关键：阶段内无提升计数清零
                    if self.accelerator.is_main_process:
                        if self.config['use_ddp']:
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)
                        self.log(f'[Epoch {epoch + 1}] 🎉 New best score! Saved model checkpoint to {self.saved_model_ckpt}')
                else:
                    stage_no_improve_eval += 1
                    if self.accelerator.is_main_process:
                        self.log(f'[Epoch {epoch + 1}] No improvement in current stage for {stage_no_improve_eval}/{self.schedule_cfg.get("switch_patience_eval", 5)} evaluations')

                # === 阶段调度：切换/终止 ===
                if self.schedule_enabled:
                    # 阶段1：guided-least，固定跑 N 个 epoch，评估只用于保存最优，不触发切换
                    if cur_stage == 'least':
                        pass  # 固定跑，见下方"epoch结束"处按 least_epochs 切

                    # 阶段2：sequential，连续 N 次评估无提升 → 切到 most
                    elif cur_stage == 'sequential':
                        if stage_no_improve_eval >= int(self.schedule_cfg.get('switch_patience_eval', 5)):
                            cur_stage = 'most'
                            stage_no_improve_eval = 0
                            _apply_stage(cur_stage)

                    # 阶段3：guided-most，连续 N 次评估无提升 → 训练结束
                    elif cur_stage == 'most':
                        if stage_no_improve_eval >= int(self.schedule_cfg.get('switch_patience_eval', 5)):
                            self.log(f'🛑 Stage "most" reached {stage_no_improve_eval} no-improve evaluations, stopping training.')
                            break

                # === 全局早停（仅在未启用调度时生效，保持旧逻辑） ===
                if (not self.schedule_enabled) and use_global_early_stop:
                    if improved:
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if self.accelerator.is_main_process:
                            self.log(f'[Epoch {epoch + 1}] No improvement for {no_improve_count}/{self.config["patience"]} evaluations')
                    if self.config['patience'] is not None and no_improve_count >= self.config['patience']:
                        self.log(f'🛑 Early stopping at epoch {epoch + 1} (after {eval_count} evaluations, {no_improve_count} without improvement)')
                        break

            # === 统计当前阶段已跑的 epoch 数，并处理 least→sequential 的固定切换 ===
            if self.schedule_enabled:
                stage_epoch_run += 1
                if cur_stage == 'least':
                    if stage_epoch_run >= int(self.schedule_cfg.get('least_epochs', 5)):
                        cur_stage = 'sequential'
                        stage_epoch_run = 0
                        stage_no_improve_eval = 0
                        _apply_stage(cur_stage)
                    
        self.log(f'Best epoch: {best_epoch}, Best val score: {best_val_score:.4f}')
        self.log(f'Training completed after {eval_count} evaluations (eval every {eval_interval} epochs)')
        
        # 🚀 修复：在训练结束前加载最佳模型权重
        if self.accelerator.is_main_process:
            self.log(f'Loading best checkpoint ({self.saved_model_ckpt}) for final test')
            state_dict = torch.load(self.saved_model_ckpt, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.to(next(self.model.parameters()).device)  # 保险：放回正确device
        
        return best_epoch, best_val_score

    def evaluate(self, dataloader, split='test'):
        """
        评估模型 - 适配diffusion模式，支持多种beam search模式
        
        Args:
            dataloader: 数据加载器
            split: 数据集分割名称
            
        Returns:
            OrderedDict: 评估结果字典
        """
        self.model.eval()

        # 获取要评估的beam search模式
        modes = self.config.get("beam_search_modes", ["confidence"])
        
        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        
        # 导入evaluator
        from .evaluator import DIFF_GRMEvaluator
        evaluator = DIFF_GRMEvaluator(self.config, self.tokenizer)
        
        for batch in val_progress_bar:
            with torch.no_grad():
                # 🚀 设置当前split，用于beam search配置选择
                self.config["current_split"] = split  # split == "val" / "test"
                
                # 对每个mode进行生成和评估
                for mode in modes:
                    # 生成序列
                    maxk = max(self.config['topk'])
                    preds = self.model.generate(batch, n_return_sequences=maxk, mode=mode)  # [B, maxk, n_digit]
                    
                    # 获取真实标签
                    labels = batch['labels']  # [B, n_digit]
                    
                    # 计算指标
                    batch_results = evaluator.calculate_metrics(preds, labels, suffix=("" if mode=="confidence" else f"_{mode}"))
                    
                    # 累积结果
                    for key, values in batch_results.items():
                        all_results[key].extend(values.tolist())

        # 计算平均指标
        final_results = OrderedDict()
        for key, values_list in all_results.items():
            final_results[key] = np.mean(values_list)

        # 🚀 打印最终统计结果
        evaluator.print_final_stats()

        self.model.train()
        return final_results

    def end(self):
        """结束训练"""
        self.accelerator.end_training()

    def log(self, message, level='info'):
        """输出日志"""
        if self.accelerator is not None:
            if self.accelerator.is_main_process:
                print(message)
        else:
            print(message) 