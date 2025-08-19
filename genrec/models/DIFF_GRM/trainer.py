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

    # ===== 阶段式训练工具 =====
    def _build_stage_plan(self):
        """
        优先读取 config['mask_schedule']['pipeline']，没有则返回空列表（保持旧行为）。
        """
        ms = self.schedule_cfg or {}
        plan = []
        if 'pipeline' in ms and ms['pipeline']:
            for st in ms['pipeline']:
                s = dict(st)  # 拷贝一份
                assert 'strategy' in s, "Each stage in pipeline must have 'strategy'"
                s.setdefault('epochs', -1)
                plan.append(s)
        return plan

    def _apply_stage(self, stage_idx):
        """切换到第 stage_idx 个阶段（调用模型的 set_masking_mode）"""
        stage = self.stage_plan[stage_idx]
        strat = stage['strategy']
        kwargs = {k: v for k, v in stage.items() if k not in ('strategy', 'epochs')}
        # 真正切换
        self.model.set_masking_mode(strat, **kwargs)

        self.cur_stage_idx = stage_idx
        self.cur_stage_epochs_done = 0

        if self.accelerator.is_main_process:
            print(f"[SCHEDULE] >>> Enter Stage #{stage_idx+1}: {strat}, args={kwargs}")

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
        best_val_score = -float("inf")  # 更鲁棒，避免指标可能为负数时被 -1 卡住
        
        # ===== 初始化阶段式训练 =====
        self.stage_plan = self._build_stage_plan()
        self.cur_stage_idx = 0
        self.cur_stage_epochs_done = 0
        if self.schedule_cfg.get('enabled', False) and self.stage_plan:
            # 可选：覆盖评估频率，便于观察阶段切换
            if self.schedule_cfg.get('eval_start_epoch_override') is not None:
                self.config['eval_start_epoch'] = int(self.schedule_cfg['eval_start_epoch_override'])
            if self.schedule_cfg.get('eval_interval_override') is not None:
                self.config['eval_interval'] = int(self.schedule_cfg['eval_interval_override'])
            self._apply_stage(0)
        
        # 新增：跟踪评估次数和无提升的评估次数
        eval_count = 0
        no_improve_count = 0
        
        # 新：若启用调度，用覆盖值；否则用原值
        eval_start_epoch = self.schedule_cfg.get('eval_start_epoch_override', self.config.get('eval_start_epoch', 1)) \
                           if self.schedule_enabled else self.config.get('eval_start_epoch', 1)
        eval_interval = self.schedule_cfg.get('eval_interval_override', self.config['eval_interval']) \
                        if self.schedule_enabled else self.config['eval_interval']

        # 全局早停：无论是否启用调度都生效
        use_global_early_stop = int(self.config.get('patience', 0) or 0) > 0
        min_delta = float(self.config.get('min_delta', 0.0))
        
        self.log(f'[TRAINING] Evaluation config: start from epoch {eval_start_epoch}, interval: {eval_interval}')
        if self.schedule_enabled and self.stage_plan:
            stage_names = [stage['strategy'] for stage in self.stage_plan]
            self.log(f'[TRAINING] Auto schedule enabled: {" → ".join(stage_names)}')

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
                improved = val_score > (best_val_score + min_delta)
                if improved:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        if self.config.get('use_ddp', False):  # 避免没配该键时报 KeyError
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)
                        self.log(f'[Epoch {epoch + 1}] 🎉 New best score! Saved model checkpoint to {self.saved_model_ckpt}')
                else:
                    if self.accelerator.is_main_process:
                        self.log(f'[Epoch {epoch + 1}] No improvement in current evaluation')

                # === 阶段调度：切换/终止 ===
                # 新的管线调度逻辑：按epoch数切换，不使用评估无提升
                pass

                # === 全局早停（无论是否启用调度都生效） ===
                if use_global_early_stop:
                    if improved:
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if self.accelerator.is_main_process:
                            self.log(f'[Epoch {epoch + 1}] No improvement for {no_improve_count}/{self.config["patience"]} evaluations (min_delta={min_delta})')
                    if no_improve_count >= int(self.config["patience"]):
                        self.log(f'🛑 Early stopping at epoch {epoch + 1} (after {eval_count} evaluations)')
                        break

            # ===== 阶段推进 =====
            if self.schedule_cfg.get('enabled', False) and self.stage_plan:
                self.cur_stage_epochs_done += 1
                cur_stage = self.stage_plan[self.cur_stage_idx]
                need_switch = (cur_stage.get('epochs', -1) > 0 and
                               self.cur_stage_epochs_done >= int(cur_stage['epochs']))
                if need_switch and (self.cur_stage_idx + 1) < len(self.stage_plan):
                    self._apply_stage(self.cur_stage_idx + 1)
                    
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