# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Dict, List, Any


def stack_to_tensor(seq, dtype=None):
    """通用工具：将所有元素stack成tensor；若传入dtype则做类型转换。"""
    if torch.is_tensor(seq[0]):
        out = torch.stack(seq, dim=0)
        return out.to(dtype) if dtype is not None else out
    return torch.tensor(seq, dtype=(dtype if dtype is not None else torch.long))


def collate_fn_train(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    训练时的collate函数
    
    Args:
        batch: 包含以下字段的字典列表：
            - history_sid: 历史SID序列 [seq_len, n_digit]
            - history_mask: 历史掩码 [seq_len]
            - decoder_input_ids: decoder输入 [n_digit]
            - decoder_labels: decoder标签 [n_digit]
    
    Returns:
        批处理后的字典
    """
    return {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),                      # [B, S, n_digit]
        'history_mask': stack_to_tensor([b['history_mask'] for b in batch], dtype=torch.bool),  # [B, S]
        'decoder_input_ids': stack_to_tensor([b['decoder_input_ids'] for b in batch]),          # [B, n_digit]
        'decoder_labels': stack_to_tensor([b['decoder_labels'] for b in batch]),                # [B, n_digit]
    }


def collate_fn_val(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    验证时的collate函数
    
    Args:
        batch: 包含以下字段的字典列表：
            - history_sid: 历史SID序列 [seq_len, n_digit]
            - history_mask: 历史掩码 [seq_len]
            - labels: 真标签序列 [n_digit]
    
    Returns:
        批处理后的字典
    """
    return {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),
        'history_mask': stack_to_tensor([b['history_mask'] for b in batch], dtype=torch.bool),
        'labels': stack_to_tensor([b['labels'] for b in batch]),
    }


def collate_fn_test(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    测试时的collate函数（与验证相同）
    """
    return collate_fn_val(batch) 