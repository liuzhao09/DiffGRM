# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Dict, List, Any


def stack_to_tensor(seq):
    """通用工具：将所有元素stack成tensor"""
    if torch.is_tensor(seq[0]):
        return torch.stack(seq, dim=0)
    return torch.tensor(seq, dtype=torch.long)


def collate_fn_train(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    训练时的collate函数
    
    Args:
        batch: 包含以下字段的字典列表：
            - history_sid: 历史SID序列 [50, 4]
            - decoder_input_ids: decoder输入 [4] - [BOS, sid0, sid1, sid2]
            - decoder_labels: decoder标签 [4] - [sid0, sid1, sid2, sid3]
    
    Returns:
        批处理后的字典
    """
    return {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),  # [B, 50, 4]
        'decoder_input_ids': stack_to_tensor([b['decoder_input_ids'] for b in batch]),  # [B, 4]
        'decoder_labels': stack_to_tensor([b['decoder_labels'] for b in batch]),  # [B, 4]
    }


def collate_fn_val(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    验证时的collate函数
    
    Args:
        batch: 包含以下字段的字典列表：
            - history_sid: 历史SID序列 [50, 4]
            - labels: 真标签序列 [4] - [sid0, sid1, sid2, sid3]
    
    Returns:
        批处理后的字典
    """
    return {
        'history_sid': stack_to_tensor([b['history_sid'] for b in batch]),  # [B, 50, 4]
        'labels': stack_to_tensor([b['labels'] for b in batch]),  # [B, 4]
    }


def collate_fn_test(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    测试时的collate函数（与验证相同）
    """
    return collate_fn_val(batch) 