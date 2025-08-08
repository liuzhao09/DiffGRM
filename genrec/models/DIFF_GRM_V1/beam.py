# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


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
            logits = out.logits.squeeze(1)  # [current_batch_size, codebook_size] - 移除序列维度
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
                    logits = model.forward_decoder_only(batch, return_loss=False, digit=digit).logits[0].squeeze(0)
                
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
                    outputs = model.forward_decoder_only(batch, return_loss=False, digit=digit)
                    logits = outputs.logits[0].squeeze(0)
                
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
                logits = outputs.logits[0].squeeze(0)
                
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


def iterative_mask_decode(model, encoder_hidden, n_return_sequences=1, tokenizer=None):
    """
    迭代式掩码填充解码，模仿TensorFlow工程代码的diffusion推理过程
    
    Args:
        model: DIFF_GRM模型
        encoder_hidden: encoder输出 [B, seq_len, emb_dim]
        n_return_sequences: 返回序列数量
        tokenizer: tokenizer对象
    
    Returns:
        generated_sequences: [B, n_return_sequences, n_digit] 生成的序列
    """
    device = encoder_hidden.device
    batch_size = encoder_hidden.size(0)
    n_digit = model.n_digit
    codebook_size = model.codebook_size
    
    # 设置beam search参数（模仿TensorFlow代码）
    pre_cut_num_list = [256, 512, 512, 512]  # 每步预选候选数
    beam_search_num_list = [256, min(512, n_return_sequences), min(512, n_return_sequences), min(512, n_return_sequences)]
    
    # 初始化：全掩码状态
    MASK_ID = tokenizer.mask_token  # -1
    current_beam_size = 1
    
    # 初始化beam序列：[B*beam_size, n_digit]，全部MASK
    beam_sequences = torch.full((batch_size * current_beam_size, n_digit), 
                               MASK_ID, device=device, dtype=torch.long)
    beam_logprobs = torch.zeros(batch_size * current_beam_size, device=device)
    
    # Step 0: 全掩码预测，获取所有位置的概率
    with torch.no_grad():
        # 构建mask_positions：全1表示全部被掩码
        mask_positions = torch.ones(batch_size, n_digit, device=device)
        
        # 构建batch
        batch_dict = {
            'decoder_input_ids': torch.zeros(batch_size, n_digit, device=device, dtype=torch.long),
            'encoder_hidden': encoder_hidden,
            'mask_positions': mask_positions
        }
        
        # 前向传播
        outputs = model.forward_decoder_only(batch_dict, digit=None)  # 获取所有位置的logits
        all_logits = outputs.logits  # [B, n_digit, codebook_size]
        
        # 计算log probabilities
        all_log_probs = F.log_softmax(all_logits, dim=-1)  # [B, n_digit, codebook_size]
        
        # 拼接所有位置的概率: [B, n_digit * codebook_size]
        flattened_log_probs = all_log_probs.view(batch_size, -1)
        
        # 选择top-k候选
        pre_cut_num = pre_cut_num_list[0]
        top_k_probs, top_k_indices = torch.topk(flattened_log_probs, k=pre_cut_num)
        
        # 解析位置和token
        digit_positions = top_k_indices // codebook_size  # 第几个digit
        token_ids = top_k_indices % codebook_size  # codebook内的ID
        
        # 更新beam数量
        current_beam_size = beam_search_num_list[0]
        
        # 选择最佳beam_search_num个候选
        beam_logprobs = top_k_probs[:, :current_beam_size].flatten()  # [B * beam_size]
        beam_digit_positions = digit_positions[:, :current_beam_size].flatten()  # [B * beam_size]
        beam_token_ids = token_ids[:, :current_beam_size].flatten()  # [B * beam_size]
        
        # 构建新的beam序列
        beam_sequences = torch.full((batch_size * current_beam_size, n_digit), 
                                   MASK_ID, device=device, dtype=torch.long)
        
        # 填充选中的位置
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(current_beam_size)
        beam_indices = torch.arange(batch_size * current_beam_size, device=device)
        
        beam_sequences[beam_indices, beam_digit_positions] = beam_token_ids
        
        # 扩展encoder_hidden以匹配beam size
        encoder_hidden_expanded = encoder_hidden.unsqueeze(1).repeat(1, current_beam_size, 1, 1)
        encoder_hidden_expanded = encoder_hidden_expanded.view(-1, encoder_hidden.size(1), encoder_hidden.size(2))
    
    # Steps 1-2: 继续填充剩余位置
    for step in range(1, 3):  # step=1,2
        pre_cut_num = pre_cut_num_list[step]
        beam_search_num = beam_search_num_list[step]
        
        with torch.no_grad():
            # 构建当前状态的mask_positions
            mask_positions = (beam_sequences == MASK_ID).float()  # [B*current_beam_size, n_digit]
            
            # 构建batch
            batch_dict = {
                'decoder_input_ids': torch.clamp(beam_sequences, min=0),  # 将MASK_ID(-1)替换为0占位
                'encoder_hidden': encoder_hidden_expanded,
                'mask_positions': mask_positions
            }
            
            # 前向传播获取所有位置logits
            outputs = model.forward_decoder_only(batch_dict, digit=None)
            all_logits = outputs.logits  # [B*current_beam_size, n_digit, codebook_size]
            
            # 计算log probabilities
            all_log_probs = F.log_softmax(all_logits, dim=-1)
            
            # 只考虑被掩码的位置
            mask_expanded = mask_positions.unsqueeze(-1).expand_as(all_log_probs)  # [B*current_beam_size, n_digit, codebook_size]
            masked_log_probs = all_log_probs * mask_expanded + (1 - mask_expanded) * (-1e4)
            
            # 拼接: [B*current_beam_size, n_digit * codebook_size]
            flattened_log_probs = masked_log_probs.view(batch_size * current_beam_size, -1)
            
            # 选择top-k候选
            top_k_probs, top_k_indices = torch.topk(flattened_log_probs, k=pre_cut_num)
            
            # 计算累积概率
            beam_logprobs_expanded = beam_logprobs.unsqueeze(-1).expand(-1, pre_cut_num)
            candidate_logprobs = beam_logprobs_expanded + top_k_probs  # [B*current_beam_size, pre_cut_num]
            
            # 选择全局最佳候选
            candidate_logprobs_flat = candidate_logprobs.view(batch_size, -1)  # [B, current_beam_size * pre_cut_num]
            
            # 选择top beam_search_num
            best_logprobs, best_indices = torch.topk(candidate_logprobs_flat, k=beam_search_num)
            
            # 解析选中的beam和位置
            parent_beam_ids = best_indices // pre_cut_num  # 父beam ID
            token_choice_ids = best_indices % pre_cut_num  # 在top_k中的选择
            
            # 获取新的序列
            new_beam_sequences = []
            new_beam_logprobs = []
            
            for b in range(batch_size):
                for k in range(beam_search_num):
                    # 获取父beam
                    parent_beam_idx = b * current_beam_size + parent_beam_ids[b, k]
                    parent_sequence = beam_sequences[parent_beam_idx].clone()
                    
                    # 获取要填充的位置和token
                    choice_idx = token_choice_ids[b, k]
                    global_idx = top_k_indices[parent_beam_idx, choice_idx]
                    digit_pos = global_idx // codebook_size
                    token_id = global_idx % codebook_size
                    
                    # 更新序列
                    parent_sequence[digit_pos] = token_id
                    new_beam_sequences.append(parent_sequence)
                    new_beam_logprobs.append(best_logprobs[b, k])
            
            # 更新beam状态
            beam_sequences = torch.stack(new_beam_sequences)  # [B * beam_search_num, n_digit]
            beam_logprobs = torch.stack(new_beam_logprobs)  # [B * beam_search_num]
            current_beam_size = beam_search_num
            
            # 更新encoder_hidden
            if current_beam_size != encoder_hidden_expanded.size(0) // batch_size:
                encoder_hidden_expanded = encoder_hidden.unsqueeze(1).repeat(1, current_beam_size, 1, 1)
                encoder_hidden_expanded = encoder_hidden_expanded.view(-1, encoder_hidden.size(1), encoder_hidden.size(2))
    
    # Step 3: 最终填充
    step = 3
    with torch.no_grad():
        # 找到剩余的掩码位置
        mask_positions = (beam_sequences == MASK_ID).float()
        
        # 对于每个beam，找到第一个被掩码的位置
        # 如果没有被掩码的位置，argmax会返回0，但我们需要检查这种情况
        remaining_positions = torch.argmax(mask_positions.float(), dim=1)  # [B * current_beam_size]
        
        # 构建batch
        batch_dict = {
            'decoder_input_ids': torch.clamp(beam_sequences, min=0),
            'encoder_hidden': encoder_hidden_expanded,
            'mask_positions': mask_positions
        }
        
        # 分别计算每个位置的logits
        final_logprobs_list = []
        
        for digit in range(n_digit):
            outputs = model.forward_decoder_only(batch_dict, digit=digit)
            digit_logits = outputs.logits  # [B*current_beam_size, 1, codebook_size]
            digit_logits = digit_logits.squeeze(1)  # [B*current_beam_size, codebook_size]
            digit_log_probs = F.log_softmax(digit_logits, dim=-1)
            
            # 只考虑需要该digit的beam
            digit_mask = (remaining_positions == digit).float()  # [B*current_beam_size]
            masked_log_probs = digit_log_probs + (1 - digit_mask.unsqueeze(-1)) * (-1e4)
            
            final_logprobs_list.append(masked_log_probs)
        
        # 合并所有digit的概率
        all_final_logprobs = torch.stack(final_logprobs_list, dim=1)  # [B*current_beam_size, n_digit, codebook_size]
        
        # 选择每个beam的最佳token
        final_sequences = beam_sequences.clone()
        final_logprobs = beam_logprobs.clone()
        
        for i in range(batch_size * current_beam_size):
            remaining_pos = remaining_positions[i].item()  # 确保是标量
            if mask_positions[i, remaining_pos] > 0:  # 该位置确实被掩码
                # 获取该位置的logits
                position_logits = all_final_logprobs[i, remaining_pos]  # [codebook_size]
                best_token_logprob, best_token_id = torch.max(position_logits, dim=0)
                final_sequences[i, remaining_pos] = best_token_id.item()  # 确保是标量
                final_logprobs[i] += best_token_logprob.item()  # 确保是标量
    
    # 重新排列为 [B, beam_size, n_digit] 格式
    final_sequences = final_sequences.view(batch_size, current_beam_size, n_digit)
    
    # 只返回前n_return_sequences个
    n_return = min(n_return_sequences, current_beam_size)
    final_sequences = final_sequences[:, :n_return, :]
    
    return final_sequences


def simple_mask_decode(model, encoder_hidden, n_return_sequences=1, tokenizer=None):
    """
    简化版的掩码解码，用于调试
    """
    device = encoder_hidden.device
    batch_size = encoder_hidden.size(0)
    n_digit = model.n_digit
    
    # 从全掩码开始
    MASK_ID = tokenizer.mask_token
    sequences = torch.full((batch_size, n_return_sequences, n_digit), 
                          MASK_ID, device=device, dtype=torch.long)
    
    # 逐个位置填充
    for digit in range(n_digit):
        with torch.no_grad():
            # 当前所有序列的mask状态
            current_sequences = sequences.view(-1, n_digit)  # [B*n_return, n_digit]
            mask_positions = (current_sequences == MASK_ID).float()
            
            # 扩展encoder_hidden
            encoder_expanded = encoder_hidden.unsqueeze(1).repeat(1, n_return_sequences, 1, 1)
            encoder_expanded = encoder_expanded.view(-1, encoder_hidden.size(1), encoder_hidden.size(2))
            
            # 构建batch
            batch_dict = {
                'decoder_input_ids': torch.clamp(current_sequences, min=0),
                'encoder_hidden': encoder_expanded,
                'mask_positions': mask_positions
            }
            
            # 预测当前digit
            outputs = model.forward_decoder_only(batch_dict, digit=digit)
            logits = outputs.logits.squeeze(1)  # [B*n_return, codebook_size] - 移除序列维度
            
            # 选择最佳token
            best_tokens = torch.argmax(logits, dim=-1)  # [B*n_return]
            
            # 更新序列
            current_sequences[:, digit] = best_tokens
            sequences = current_sequences.view(batch_size, n_return_sequences, n_digit)
    
    return sequences 