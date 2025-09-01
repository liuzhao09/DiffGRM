import torch
from typing import Tuple


@torch.no_grad()
def _topk_2d_sum(s: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    给定两列的候选得分做两两相加并取整体 top-k。

    Args:
        s: [B, X, Y]，X/Y分别为两列的候选数
        k: 需要的前k个组合

    Returns:
        values: [B, k]
        ix: [B, k]，第一列的索引
        iy: [B, k]，第二列的索引
    """
    B, X, Y = s.shape
    flat = s.reshape(B, -1)
    top_k = min(k, X * Y)
    vals, idx = torch.topk(flat, k=top_k, dim=-1)
    ix = idx // Y
    iy = idx % Y
    return vals, ix, iy


@torch.no_grad()
def combine_remaining_topk(
    per_digit_logp: torch.Tensor,
    topK_final: int,
    per_digit_topL: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 r 个待填列的 log 概率进行组合 top-k：
    - 每列先取 topL
    - 逐列两两合并，每次都截断到 topK_final，直到合并完所有列

    Args:
        per_digit_logp: [B, r, V]，r为剩余列数，V为codebook大小
        topK_final: 组合后保留的候选数
        per_digit_topL: 每列先行截断的候选数（<= V）

    Returns:
        comb_vals: [B, topK_final] 组合后的总logp
        comb_tokens: [B, topK_final, r] 各列选到的 codebook id
    """
    B, r, V = per_digit_logp.shape
    assert r >= 1
    L = min(per_digit_topL, V)

    # 各列先取 topL
    vals = []
    ids = []
    for i in range(r):
        v, idx = torch.topk(per_digit_logp[:, i], k=L, dim=-1)  # [B, L]
        vals.append(v)
        ids.append(idx)  # codebook ids

    # 逐列两两合并
    cur_vals = vals[0]  # [B, L]
    cur_ids = ids[0].unsqueeze(-1)  # [B, L, 1]
    for i in range(1, r):
        s = cur_vals.unsqueeze(2) + vals[i].unsqueeze(1)  # [B, L, L]
        v, ix, iy = _topk_2d_sum(s, k=min(topK_final, L * L))  # [B, K]

        # 回溯 tokens
        prev = torch.gather(cur_ids, dim=1, index=ix.unsqueeze(-1).expand(-1, -1, cur_ids.shape[-1]))  # [B, K, i]
        tok_i = torch.gather(ids[i], dim=1, index=iy)  # [B, K]
        cur_ids = torch.cat([prev, tok_i.unsqueeze(-1)], dim=-1)  # [B, K, i+1]
        cur_vals = v

    return cur_vals, cur_ids

