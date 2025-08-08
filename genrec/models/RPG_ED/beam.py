# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class BeamHypothesis:
    """Beam搜索假设"""
    def __init__(self, tokens: List[int], score: float):
        self.tokens = tokens
        self.score = score

    def __lt__(self, other):
        return self.score > other.score  # 用于heapq，分数高的在前


def beam_search_batch(model, encoder_hidden, beam_size=10, max_len=4, tokenizer=None):
    """
    大规模向量化Beam Search - 参照工程代码实现
    
    Args:
        model: RPG_ED模型
        encoder_hidden: Encoder输出 [batch_size, seq_len, hidden_dim]
        beam_size: 最终beam大小
        max_len: 最大生成长度
        tokenizer: Tokenizer
    
    Returns:
        torch.Tensor: 生成的token序列 [batch_size, beam_size, max_len]
    """
    device = encoder_hidden.device
    batch_size = encoder_hidden.size(0)
    
    # 参照工程代码的beam search配置
    pre_cut_num_list = [256, 256, 256, 256]        # 每步预选候选数
    beam_search_num_list = [256, min(512, beam_size), min(512, beam_size), min(512, beam_size)]  # 每步保留beam数
    
    # -------- 0. 初始化 --------
    # 从BOS开始
    current_beam_size = 1
    beam_tokens = torch.full((batch_size, 1, 0), 0, device=device, dtype=torch.long)  # 空序列
    beam_logits = torch.zeros(batch_size, 1, device=device)  # 累积log-prob
    ori_decoder_inputs = torch.full((batch_size * 1, 1), tokenizer.bos_token, device=device, dtype=torch.long)
    
    # -------- 1. 逐步解码 --------
    for step in range(max_len):
        pre_cut_num = pre_cut_num_list[step]
        beam_search_num = beam_search_num_list[step]
        
        # 当前beam数量
        current_batch_size = batch_size * current_beam_size
        
        # 通过decoder
        batch_dict = {
            'decoder_input_ids': ori_decoder_inputs,
            'encoder_hidden': encoder_hidden.unsqueeze(1).repeat(1, current_beam_size, 1, 1).view(current_batch_size, -1, encoder_hidden.size(-1))
        }
        
        with torch.no_grad():
            out = model.forward_decoder_only(batch_dict, return_loss=False, digit=step)
            logits = out.logits[:, -1, :]  # (current_batch_size, 256) - 只要最后一个位置
            probs = torch.softmax(logits, dim=-1)
        
        # 选择top-k候选
        topk_probs, topk_indices = torch.topk(probs, k=pre_cut_num, dim=-1)  # (current_batch_size, pre_cut_num)
        topk_log_probs = torch.log(topk_probs + 1e-10)  # 避免log(0)
        
        # Reshape为beam维度
        topk_log_probs = topk_log_probs.view(batch_size, current_beam_size, pre_cut_num)  # (B, current_beam, pre_cut)
        topk_indices = topk_indices.view(batch_size, current_beam_size, pre_cut_num)  # (B, current_beam, pre_cut)
        
        # 计算累积分数
        beam_logits_expanded = beam_logits.unsqueeze(-1).expand(-1, -1, pre_cut_num)  # (B, current_beam, pre_cut)
        candidate_scores = beam_logits_expanded + topk_log_probs  # (B, current_beam, pre_cut)
        
        # 选择最佳beam_search_num个候选
        candidate_scores_flat = candidate_scores.view(batch_size, -1)  # (B, current_beam * pre_cut)
        topk_indices_flat = topk_indices.view(batch_size, -1)  # (B, current_beam * pre_cut)
        
        best_scores, best_indices = torch.topk(candidate_scores_flat, k=min(beam_search_num, candidate_scores_flat.size(-1)), dim=-1)  # (B, beam_search_num)
        
        # 确定选中的beam和token
        selected_beam_ids = best_indices // pre_cut_num  # (B, beam_search_num)
        selected_token_ids = best_indices % pre_cut_num  # (B, beam_search_num)
        
        # 获取选中的token（codebook ID，0-255）
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, beam_search_num)
        selected_tokens = topk_indices_flat[batch_indices, best_indices]  # (B, beam_search_num)
        
        # 转换为token ID用于embedding
        selected_token_embeddings = selected_tokens + tokenizer.sid_offset + step * tokenizer.codebook_size
        
        # 更新beam序列
        if step == 0:
            # 第一步：从空序列开始
            new_beam_tokens = selected_tokens.unsqueeze(-1)  # (B, beam_search_num, 1)
        else:
            # 后续步骤：扩展现有序列
            # 选择对应的父beam
            selected_parent_tokens = torch.gather(beam_tokens, 1, 
                                                selected_beam_ids.unsqueeze(-1).expand(-1, -1, beam_tokens.size(-1)))  # (B, beam_search_num, step)
            new_beam_tokens = torch.cat([selected_parent_tokens, selected_tokens.unsqueeze(-1)], dim=-1)  # (B, beam_search_num, step+1)
        
        # 更新状态
        beam_tokens = new_beam_tokens
        beam_logits = best_scores
        current_beam_size = beam_search_num
        
        # 准备下一步的decoder输入
        if step < max_len - 1:  # 不是最后一步
            # 构建完整的decoder输入序列：[BOS] + 已生成的tokens
            bos_tokens = torch.full((batch_size, beam_search_num, 1), tokenizer.bos_token, device=device, dtype=torch.long)
            
            # 将已生成的codebook ID转换为token ID
            generated_token_ids = torch.zeros_like(beam_tokens)
            for d in range(step + 1):
                generated_token_ids[:, :, d] = beam_tokens[:, :, d] + tokenizer.sid_offset + d * tokenizer.codebook_size
            
            decoder_input_sequence = torch.cat([bos_tokens, generated_token_ids], dim=-1)  # (B, beam_search_num, step+2)
            ori_decoder_inputs = decoder_input_sequence.view(batch_size * beam_search_num, -1)
    
    # -------- 2. 输出 --------
    # 如果最终beam数量超过要求，截取前beam_size个
    final_beam_size = min(beam_size, beam_tokens.size(1))
    beam_tokens = beam_tokens[:, :final_beam_size, :]
    
    return beam_tokens  # (B, final_beam_size, max_len)


def beam_generate(model, encoder_hidden, beam_size=10, max_len=4, tokenizer=None, n_best=1):
    """
    Beam search生成器
    
    Args:
        model: RPG_ED模型
        encoder_hidden: Encoder输出 [batch_size, seq_len, hidden_dim]
        beam_size: Beam大小
        max_len: 最大生成长度
        tokenizer: Tokenizer
        n_best: 返回的最佳序列数量
    
    Returns:
        List[List[List[int]]]: 生成的token序列 [batch_size, n_best, seq_len]
    """
    batch_size = encoder_hidden.size(0)
    device = encoder_hidden.device
    
    # 初始化beam
    beams = [[BeamHypothesis([tokenizer.bos_token], 0.0)] for _ in range(batch_size)]
    
    # 逐token展开
    for _ in range(max_len):
        new_beams = []
        
        for b_idx in range(batch_size):
            cur_beams = beams[b_idx]
            cand_beams = []
            
            for bh in cur_beams:
                if len(bh.tokens) - 1 == max_len:  # 已生成max_len个有效token
                    cand_beams.append(bh)
                    continue
                
                # Decoder forward
                input_ids = torch.tensor([bh.tokens], device=device)
                batch = {
                    'decoder_input_ids': input_ids,
                    'encoder_hidden': encoder_hidden[b_idx:b_idx+1]
                }
                
                with torch.no_grad():
                    # 修复：传入digit参数，当前是第几个token（去掉BOS）
                    digit = len(bh.tokens) - 1  # 当前digit位置
                    logits = model.forward_decoder_only(batch, return_loss=False, digit=digit).logits[0, -1]
                
                topk_logp, topk_idx = torch.topk(logits, beam_size)  # beam_size即保活数
                
                logp = F.log_softmax(logits, dim=-1)
                for idx in topk_idx:
                    # 修复：将codebook ID转换为token ID（加上offset）
                    token_id = idx.item()
                    digit = len(bh.tokens) - 1  # 当前digit位置
                    # 修复：使用tokenizer.codebook_size而不是硬编码的256
                    token_id += tokenizer.sid_offset + digit * tokenizer.codebook_size
                    
                    new_tokens = bh.tokens + [token_id]
                    new_score = bh.score + logp[idx].item()
                    cand_beams.append(BeamHypothesis(new_tokens, new_score))
            
            # 按得分排序并保留beam_size
            cand_beams.sort()
            new_beams.append(cand_beams[:beam_size])
        
        beams = new_beams
    
    # 组装n_best结果
    results = []
    for sample_beams in beams:
        sample_beams.sort()  # 已按score排过，但保险再排
        seqs = [bh.tokens[1:] for bh in sample_beams[:n_best]]  # 去掉BOS
        
        # beam_size可能<n_best，用PAD序列补齐方便reshape
        pad_seq = [tokenizer.pad_token] * max_len
        while len(seqs) < n_best:
            seqs.append(pad_seq)
        results.append(seqs)  # [n_best, max_len]
    
    return results  # [B, n_best, max_len]


def beam_generate_with_cache(model, encoder_hidden, beam_size=10, max_len=4, tokenizer=None):
    """
    带KV cache的beam search生成器（优化版本）
    """
    batch_size = encoder_hidden.size(0)
    device = encoder_hidden.device
    
    # 初始化KV cache
    kv_cache = {}
    for layer_idx in range(len(model.decoder_blocks)):
        kv_cache[layer_idx] = None
    
    # 初始化beam
    beams = []
    for i in range(batch_size):
        sample_beams = [BeamHypothesis([tokenizer.bos_token], 0.0)]
        beams.append(sample_beams)
    
    # 逐token生成
    for step in range(max_len):
        new_beams = []
        
        for batch_idx in range(batch_size):
            sample_beams = beams[batch_idx]
            new_sample_beams = []
            
            for beam in sample_beams:
                if len(beam.tokens) >= max_len:
                    new_sample_beams.append(beam)
                    continue
                
                # 准备输入
                input_ids = torch.tensor([beam.tokens], device=device)
                
                # 创建batch
                batch = {
                    'decoder_input_ids': input_ids,
                    'encoder_hidden': encoder_hidden[batch_idx:batch_idx+1]
                }
                
                # 前向传播（使用KV cache）
                with torch.no_grad():
                    # 修复：传入digit参数，当前是第几个token（去掉BOS）
                    digit = len(beam.tokens) - 1  # 当前digit位置
                    outputs = model.forward_decoder_with_cache(batch, kv_cache, return_loss=False, digit=digit)
                    logits = outputs.logits[0, -1, :]
                
                # 获取top-k候选
                top_k_logits, top_k_indices = torch.topk(logits, beam_size * 2)
                
                # 扩展beam
                for i in range(min(beam_size * 2, len(top_k_indices))):
                    token_id = top_k_indices[i].item()
                    # 修复：将codebook ID转换为token ID（加上offset）
                    digit = len(beam.tokens) - 1  # 当前digit位置
                    # 修复：使用tokenizer.codebook_size而不是硬编码的256
                    token_id += tokenizer.sid_offset + digit * tokenizer.codebook_size
                    
                    log_prob = F.log_softmax(logits, dim=-1)[top_k_indices[i]].item()
                    
                    new_tokens = beam.tokens + [token_id]
                    new_score = beam.score + log_prob
                    
                    new_beam = BeamHypothesis(new_tokens, new_score)
                    new_sample_beams.append(new_beam)
            
            # 选择top beam_size个
            new_sample_beams.sort()
            new_sample_beams = new_sample_beams[:beam_size]
            new_beams.append(new_sample_beams)
        
        beams = new_beams
    
    # 返回最佳结果
    results = []
    for batch_idx in range(batch_size):
        best_beam = beams[batch_idx][0]
        tokens = best_beam.tokens[1:]  # 移除BOS token
        results.append(tokens)
    
    return results


def greedy_generate(model, encoder_hidden, max_len=4, tokenizer=None):
    """
    贪心生成器（用于调试）
    """
    batch_size = encoder_hidden.size(0)
    device = encoder_hidden.device
    
    results = []
    
    for batch_idx in range(batch_size):
        tokens = [tokenizer.bos_token]
        
        for step in range(max_len):
            input_ids = torch.tensor([tokens], device=device)
            
            batch = {
                'decoder_input_ids': input_ids,
                'encoder_hidden': encoder_hidden[batch_idx:batch_idx+1]
            }
            
            with torch.no_grad():
                # 修复：传入digit参数，当前是第几个token（去掉BOS）
                digit = len(tokens) - 1  # 当前digit位置
                outputs = model.forward_decoder_only(batch, return_loss=False, digit=digit)
                logits = outputs.logits[0, -1, :]
                
                # 选择概率最高的token
                next_token = torch.argmax(logits).item()
                # 修复：将codebook ID转换为token ID（加上offset）
                digit = len(tokens) - 1  # 当前digit位置
                # 修复：使用tokenizer.codebook_size而不是硬编码的256
                next_token += tokenizer.sid_offset + digit * tokenizer.codebook_size
                
                tokens.append(next_token)
                
                # 如果生成了EOS，停止生成
                if next_token == tokenizer.eos_token:
                    break
        
        # 移除BOS token
        tokens = tokens[1:]
        results.append(tokens)
    
    return results 