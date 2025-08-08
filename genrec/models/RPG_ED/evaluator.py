# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class RPG_EDEvaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }

        self.pad_token = self.tokenizer.pad_token
        self.maxk = max(config['topk'])
        
        # 调试信息：确认使用了RPG_EDEvaluator
        print(f'>> Using evaluator = {self.__class__.__name__}')

    def calculate_pos_index(self, preds, labels):
        """
        计算预测结果与真标签的匹配情况
        
        Args:
            preds: (batch_size, beam_size, n_digit) - 生成的SID token序列
            labels: (batch_size, n_digit) - 真标签序列 [sid0, sid1, sid2, sid3]
        
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
            print(f'DEBUG: vocab_size = {self.tokenizer.vocab_size}')
            print(f'DEBUG: sid_offset = {self.tokenizer.sid_offset}')
            print(f'DEBUG: codebook_size = {self.tokenizer.codebook_size}')
            self._debug_printed = True
        
        for i in range(batch_size):
            # 获取真标签（现在没有PAD了）
            cur_label = labels[i].tolist()
            
            for j in range(beam_size):
                # 获取预测的SID序列
                cur_pred = preds[i, j].tolist()
                
                # 修复：新的beam search直接返回codebook ID，无需转换
                cur_pred_codebook = cur_pred  # 直接使用，已经是codebook ID（0-255）
                
                # 添加调试信息
                if i == 0 and j == 0 and not hasattr(self, '_conversion_debug_printed'):
                    print(f'DEBUG: 大规模beam search结果（直接codebook ID）:')
                    print(f'  真标签: {cur_label}')
                    print(f'  预测结果: {cur_pred_codebook}')
                    print(f'  是否匹配: {cur_pred_codebook == cur_label}')
                    self._conversion_debug_printed = True
                
                # 检查是否匹配（现在都是0-255范围的codebook ID）
                if cur_pred_codebook == cur_label:
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

    def calculate_metrics(self, preds, labels):
        """计算所有指标"""
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        
        return results 