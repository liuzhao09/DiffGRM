# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


class DIFF_GRM_V1Evaluator:
    """
    DIFF_GRM模型的评估器
    """
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }

        self.pad_token = self.tokenizer.pad_token
        self.maxk = max(config['topk'])
        
        # 调试信息：确认使用了DIFF_GRM_V1Evaluator
        print(f'>> Using evaluator = {self.__class__.__name__}')

    def calculate_pos_index(self, preds, labels):
        """
        计算预测结果与真标签的匹配情况
        
        Args:
            preds: (batch_size, beam_size, n_digit) - 生成的SID序列
            labels: (batch_size, n_digit) - 真标签序列
        
        Returns:
            pos_index: (batch_size, beam_size) - 每个beam是否匹配真标签
        """
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        
        batch_size, beam_size, n_digit = preds.shape
        pos_index = torch.zeros((batch_size, beam_size), dtype=torch.bool)
        
        # 调试信息：检查第一个batch的预测和标签
        if not hasattr(self, '_debug_printed'):
            print(f'DEBUG: preds shape = {preds.shape}, labels shape = {labels.shape}')
            print(f'DEBUG: preds[0, 0] = {preds[0, 0].tolist()}')
            print(f'DEBUG: labels[0] = {labels[0].tolist()}')
            self._debug_printed = True
        
        for i in range(batch_size):
            # 获取真标签
            cur_label = labels[i].tolist()  # [n_digit]
            
            for j in range(beam_size):
                # 获取预测的SID序列
                cur_pred = preds[i, j].tolist()  # [n_digit]
                
                # 直接比较codebook IDs（0-255范围）
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    # 注意：不要break，因为我们需要标记所有匹配的beam
        
        return pos_index

    def recall_at_k(self, pos_index, k):
        """计算Recall@k"""
        # pos_index: (batch_size, beam_size)
        # 对于每个样本，检查top-k中是否有匹配
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        """计算NDCG@k（实际返回DCG@k，与通用evaluator保持一致）"""
        # pos_index: (batch_size, beam_size)
        batch_size, beam_size = pos_index.shape
        
        # 创建rank权重
        ranks = torch.arange(1, beam_size + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        
        # 对于每个样本，计算DCG
        dcg_scores = torch.where(pos_index, dcg, 0)
        dcg_at_k = dcg_scores[:, :k].sum(dim=1)
        
        # 修正：与通用evaluator保持一致，直接返回DCG@k
        # 对于单标签推荐任务，这样做是合理的，因为：
        # 1. IDCG始终等于1（理想情况下真值排第1位）
        # 2. 因此NDCG在数值上等于DCG
        # 3. 保持不同evaluator行为一致
        return dcg_at_k.cpu().float()

    def calculate_weighted_score(self, preds, labels):
        """
        计算加权综合分数：NDCG@10 * 0.6 + RECALL@10 * 0.4
        
        Args:
            preds: (batch_size, beam_size, n_digit) - 生成的SID序列
            labels: (batch_size, n_digit) - 真标签序列
        
        Returns:
            weighted_score: 加权综合分数
        """
        pos_index = self.calculate_pos_index(preds, labels)
        
        # 计算NDCG@10
        ndcg_10 = self.ndcg_at_k(pos_index, k=10)
        
        # 计算RECALL@10
        recall_10 = self.recall_at_k(pos_index, k=10)
        
        # 计算加权分数：NDCG@10 * 0.6 + RECALL@10 * 0.4
        weighted_score = 0.6 * ndcg_10 + 0.4 * recall_10
        
        return weighted_score

    def calculate_metrics(self, preds, labels):
        """计算所有指标"""
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        
        # 添加加权综合分数
        weighted_score = self.calculate_weighted_score(preds, labels)
        results['weighted_score'] = weighted_score
        
        return results 