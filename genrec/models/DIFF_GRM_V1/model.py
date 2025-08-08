# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from genrec.model import AbstractModel
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, n_head, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        assert emb_dim % n_head == 0
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_head

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(resid_drop)

        # Initialize weights
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x, attention_mask=None, key_value=None, past_key_value=None, use_cache=False, is_decoder_self_attn=False):
        B, T, C = x.size()

        if key_value is not None:
            # Cross attention: Q from x, K,V from key_value
            q = self.qkv(x)[:, :, :self.emb_dim]  # Only take Q part
            k, v = key_value.chunk(2, dim=-1)  # key_value should be [B, T_enc, 2*emb_dim]
            T_kv = k.size(1)
        else:
            # Self attention
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            T_kv = T

        # Handle past key-value cache for incremental decoding
        if past_key_value is not None and use_cache:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            T_kv = k.size(1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_kv, head_dim)
        v = v.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_kv, head_dim)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_head, T, T_kv)

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T, T_kv) or (B, 1, T, T_kv)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # Add head dimension
            att = att.masked_fill(attention_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = torch.matmul(att, v)  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, emb_dim)

        # Output projection
        y = self.resid_dropout(self.proj(y))

        # Prepare cache for next iteration
        present_key_value = (k, v) if use_cache else None

        return y, present_key_value


class FeedForward(nn.Module):

    def __init__(self, emb_dim, n_inner, resid_drop=0.1, act='gelu'):
        super().__init__()
        self.c_fc = nn.Linear(emb_dim, n_inner)
        self.c_proj = nn.Linear(n_inner, emb_dim)
        self.dropout = nn.Dropout(resid_drop)
        self.act = F.gelu if act == 'gelu' else F.relu

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):

    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1, 
                 act='gelu', layer_norm_epsilon=1e-5):
        super().__init__()
        self.ln_1 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self, x, attention_mask=None):
        # 自注意力 + 残差连接（非decoder自注意力）
        attn_output, _ = self.attn(self.ln_1(x), attention_mask=attention_mask, is_decoder_self_attn=False)
        x = x + attn_output
        
        # 前馈网络 + 残差连接
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):

    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1, 
                 act='gelu', layer_norm_epsilon=1e-5):
        super().__init__()
        self.ln_1 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.self_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.cross_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_3 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self, x, encoder_hidden=None, attention_mask=None, 
                past_key_value=None, use_cache=False, cross_key_value=None):
        # 修改：去除因果掩码，因为diffusion模型不需要严格的序列顺序
        # 自注意力（不使用因果掩码）
        attn_output, present_key_value = self.self_attn(
            self.ln_1(x), 
            attention_mask=None,  # 不使用因果掩码
            past_key_value=past_key_value[0] if past_key_value else None,
            use_cache=use_cache,
            is_decoder_self_attn=True
        )
        x = x + attn_output

        # 交叉注意力
        if encoder_hidden is not None:
            if cross_key_value is None:
                # 第一次计算或不使用cache
                encoder_kv = torch.cat([encoder_hidden, encoder_hidden], dim=-1)  # Concat K and V
            else:
                encoder_kv = cross_key_value
            
            cross_attn_output, cross_present = self.cross_attn(
                self.ln_2(x),
                key_value=encoder_kv,
                past_key_value=past_key_value[1] if past_key_value else None,
                use_cache=use_cache
            )
            x = x + cross_attn_output
            
            if use_cache:
                present_key_value = (present_key_value, cross_present)
        
        # 前馈网络
        x = x + self.mlp(self.ln_3(x))
        
        return_dict = {}
        return_dict['hidden_states'] = x
        if use_cache:
            return_dict['present_key_value'] = present_key_value
        
        return return_dict


class ModelOutput:

    def __init__(self):
        self.loss = None
        self.logits = None
        self.hidden_states = None
        self.past_key_values = None


class DIFF_GRM_V1(AbstractModel):

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super().__init__(config, dataset, tokenizer)
        
        self.config = config
        self.tokenizer = tokenizer
        self.n_digit = config['n_digit']
        self.codebook_size = config['codebook_size']
        self.vocab_size = tokenizer.vocab_size
        
        # Model dimensions
        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.n_inner = config['n_inner']
        self.dropout = config['dropout']
        
        # Encoder layers
        self.encoder_n_layer = config['encoder_n_layer']
        self.decoder_n_layer = config['decoder_n_layer']
        
        # Diffusion specific parameters - 多概率掩码配置
        if 'mask_probs' in config and config['mask_probs'] is not None:
            # 新方式：直接指定多个掩码概率
            mask_probs_raw = config['mask_probs']
            
            if isinstance(mask_probs_raw, str):
                # 字符串格式："1.0,0.75,0.5,0.25"
                self.mask_probs = [float(p.strip()) for p in mask_probs_raw.split(',')]
            elif isinstance(mask_probs_raw, (list, tuple)):
                # 列表或元组格式：[1.0, 0.75, 0.5, 0.25]
                self.mask_probs = [float(p) for p in mask_probs_raw]
            elif isinstance(mask_probs_raw, (int, float)):
                # 单个数值，转换为单元素列表
                self.mask_probs = [float(mask_probs_raw)]
            else:
                # 其他类型，尝试转换为字符串再解析
                try:
                    mask_probs_str = str(mask_probs_raw)
                    self.mask_probs = [float(p.strip()) for p in mask_probs_str.split(',')]
                except (ValueError, AttributeError):
                    raise ValueError(f"Cannot parse mask_probs: {mask_probs_raw} (type: {type(mask_probs_raw)}). "
                                   "Expected string like '1.0,0.75,0.5,0.25' or list like [1.0, 0.75, 0.5, 0.25]")
            
            self.augment_factor = len(self.mask_probs)  # 自动设置增强倍数
            print(f"[MODEL] Using multi-probability masking: {self.mask_probs}")
        else:
            # 旧方式：单一掩码概率 + 增强倍数
            mask_prob = config.get('mask_prob', 0.5)
            self.augment_factor = config.get('augment_factor', 4)
            self.mask_probs = [float(mask_prob)] * self.augment_factor  # 重复相同概率
            print(f"[MODEL] Using single-probability masking: {mask_prob} x {self.augment_factor}")
        
        # 验证掩码概率的有效性
        for i, prob in enumerate(self.mask_probs):
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"mask_probs[{i}] = {prob} is not in valid range [0.0, 1.0]")
        
        # Embeddings - 共享的 embedding table
        self.embedding = nn.Embedding(self.vocab_size, self.n_embd)
        
        # 添加与RPG_ED一致的item_mlp：将4个SID token压缩为1个token
        self.item_mlp = nn.Sequential(
            nn.Linear(4 * self.n_embd, self.n_embd),  # 4×d → d
            nn.ReLU(),
            nn.Linear(self.n_embd, self.n_embd)
        )
        
        # 新增：掩码嵌入表，用于表示被掩码的位置
        self.mask_emb_table = nn.Embedding(self.n_digit, self.n_embd)
        
        # 位置编码：只为encoder添加绝对位置编码（与RPG_ED一致）
        self.max_history_len = config.get('max_history_len', 50)  # 从config读取，默认50
        self.pos_emb_enc = nn.Embedding(self.max_history_len, self.n_embd)
        # 移除decoder位置编码，decoder只使用掩码
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                self.n_embd, self.n_head, self.n_inner,
                config['attn_pdrop'], config['resid_pdrop']
            )
            for _ in range(self.encoder_n_layer)
        ])
        
        # Decoder blocks  
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                self.n_embd, self.n_head, self.n_inner,
                config['attn_pdrop'], config['resid_pdrop']
            )
            for _ in range(self.decoder_n_layer)
        ])
        
        # Layer normalization
        self.ln_f = nn.LayerNorm(self.n_embd)
        
        # 移除独立的 codebook_heads，改用 embedding table 权重
        # self.codebook_heads = nn.ModuleList([
        #     nn.Linear(self.n_embd, self.codebook_size, bias=False)
        #     for _ in range(self.n_digit)
        # ])
        
        # Dropout
        self.drop = nn.Dropout(self.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _compute_digit_logits(self, hidden_last, digit):
        """
        使用 embedding table 权重计算指定 digit 位置的 logits
        
        Args:
            hidden_last: [B, emb_dim] decoder 输出
            digit: 指定的 digit 位置 (0-3)
        
        Returns:
            logits: [B, codebook_size] 该 digit 在对应 codebook 上的 logits
        """
        if digit is None:
            raise ValueError("digit参数不能为None，必须指定要计算的codebook位置")
        
        if digit >= self.n_digit:
            raise ValueError(f"digit={digit} 超出范围，应该在 [0, {self.n_digit-1}]")
        
        # 获取 embedding table 权重并重塑为 [n_digit, codebook_size, emb_dim]
        # vocab_size = n_digit * codebook_size，所以可以重塑
        embedding_weight = self.embedding.weight  # [vocab_size, emb_dim]
        
        # 只取 SID 部分的权重（跳过特殊 token）
        sid_start = self.tokenizer.sid_offset
        sid_end = sid_start + self.n_digit * self.codebook_size
        sid_embedding_weight = embedding_weight[sid_start:sid_end, :]  # [n_digit * codebook_size, emb_dim]
        
        # 重塑为 [n_digit, codebook_size, emb_dim]
        codebook_weights = sid_embedding_weight.view(self.n_digit, self.codebook_size, self.n_embd)
        
        # 获取指定 digit 的权重
        digit_weight = codebook_weights[digit]  # [codebook_size, emb_dim]
        
        # 计算 logits: hidden_last @ digit_weight.T
        logits = torch.matmul(hidden_last, digit_weight.t())  # [B, codebook_size]
        
        return logits

    def _compute_all_digit_logits(self, decoder_hidden):
        """
        使用 embedding table 权重计算所有 digit 位置的 logits
        
        Args:
            decoder_hidden: [B, n_digit, emb_dim] decoder 输出
        
        Returns:
            logits: [B, n_digit, codebook_size] 所有 digit 在对应 codebook 上的 logits
        """
        B, n_digit, emb_dim = decoder_hidden.shape
        
        # 获取 embedding table 权重并重塑为 [n_digit, codebook_size, emb_dim]
        embedding_weight = self.embedding.weight  # [vocab_size, emb_dim]
        
        # 只取 SID 部分的权重（跳过特殊 token）
        sid_start = self.tokenizer.sid_offset
        sid_end = sid_start + self.n_digit * self.codebook_size
        sid_embedding_weight = embedding_weight[sid_start:sid_end, :]  # [n_digit * codebook_size, emb_dim]
        
        # 重塑为 [n_digit, codebook_size, emb_dim]
        codebook_weights = sid_embedding_weight.view(self.n_digit, self.codebook_size, self.n_embd)
        
        # 批量计算所有 digit 的 logits
        # decoder_hidden: [B, n_digit, emb_dim]
        # codebook_weights: [n_digit, codebook_size, emb_dim]
        # 结果: [B, n_digit, codebook_size]
        logits = torch.einsum('bde,dce->bdc', decoder_hidden, codebook_weights)
        
        return logits

    @property
    def n_parameters(self) -> str:
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return f"{n_params:,}"

    def forward(self, batch: dict, return_loss=True) -> ModelOutput:
        """
        Diffusion训练：处理掩码数据，预测被掩码的位置
        
        Args:
            batch: 包含以下字段的字典：
                - history_sid: 历史SID序列 [B, seq_len, n_digit]
                - decoder_input_ids: decoder输入 [B, n_digit] 
                - decoder_labels: 真实标签 [B, n_digit]
        """
        device = next(self.parameters()).device
        
        # 添加调试信息
        if hasattr(self, '_debug_printed'):
            pass
        else:
            print(f"[DIFF_GRM] Using RPG_ED-style encoder: MLP compression + fixed 50-length sequence")
            print(f"[DIFF_GRM] vocab_size: {self.vocab_size}, codebook_size: {self.codebook_size}")
            print(f"[DIFF_GRM] mask_probs: {self.mask_probs}")
            self._debug_printed = True
        
        # --- Encoder ---
        history_sid = batch['history_sid'].to(device)  # [B, seq_len, n_digit]
        B, seq_len, n_digit = history_sid.shape
        
        # 与RPG_ED保持一致的处理方式
        # 1. 将history SID转换为token IDs
        history_tokens = torch.zeros(B, seq_len, n_digit, dtype=torch.long, device=device)
        for d in range(n_digit):
            # 转换为token IDs，添加sid_offset和digit偏移
            token_ids = history_sid[:, :, d] + self.tokenizer.sid_offset + d * self.codebook_size
            # 确保token ID在有效范围内
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
            history_tokens[:, :, d] = token_ids
        
        # 2. 获取token嵌入
        tok_emb = self.embedding(history_tokens)  # [B, seq_len, 4, d]
        B, S, _, d = tok_emb.shape
        
        # 3. 重塑并通过MLP压缩：4个SID token → 1个item token
        item_emb = tok_emb.reshape(B, S, 4 * d)  # [B, S, 4*d]
        item_emb = self.item_mlp(item_emb)  # [B, S, d]
        
        # 4. 添加位置编码（与RPG_ED一致）
        pos_ids = torch.arange(S, device=item_emb.device)  # (S,)
        pos_emb = self.pos_emb_enc(pos_ids)  # (S, d)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, S, d)
        
        # 5. 将位置编码加到item_emb上
        encoder_hidden = item_emb + pos_emb  # [B, S, d]
        encoder_hidden = self.drop(encoder_hidden)
        
        # Pass through encoder blocks
        encoder_hidden = encoder_hidden
        for block in self.encoder_blocks:
            encoder_hidden = block(encoder_hidden)
        
        encoder_hidden = self.ln_f(encoder_hidden)  # [B, seq_len*n_digit, emb_dim]
        
        if not return_loss:
            # 推理模式，直接返回encoder输出
            output = ModelOutput()
            output.hidden_states = encoder_hidden
            return output
        
        # --- 多概率掩码扩展 ---
        decoder_input_ids = batch['decoder_input_ids'].to(device)  # [B, n_digit]
        decoder_labels = batch['decoder_labels'].to(device)  # [B, n_digit]
        
        # 确保decoder输入在有效范围内
        decoder_input_ids = torch.clamp(decoder_input_ids, 0, self.codebook_size - 1)
        decoder_labels = torch.clamp(decoder_labels, 0, self.codebook_size - 1)
        
        # 扩展样本：每个原始样本生成多个具有不同掩码概率的版本
        all_masked_input_ids = []
        all_labels = []
        all_mask_positions = []
        all_encoder_hidden = []
        
        for view_idx, mask_prob in enumerate(self.mask_probs):
            
            # 为当前掩码概率生成掩码
            mask_positions = torch.rand(B, self.n_digit, device=device) < mask_prob  # [B, n_digit]
            
            # 确保每个样本至少有一个位置被掩码
            no_mask_samples = ~mask_positions.any(dim=1)  # [B]
            if no_mask_samples.any():
                # 对于没有掩码的样本，强制掩码第一个位置
                mask_positions[no_mask_samples, 0] = True
            
            # 应用掩码：被掩码的位置设为0
            masked_input_ids = decoder_input_ids.clone()  # [B, n_digit]
            masked_input_ids[mask_positions] = 0
            
            # 存储当前视图的数据
            all_masked_input_ids.append(masked_input_ids)
            all_labels.append(decoder_labels)  # 标签保持不变
            all_mask_positions.append(mask_positions.float())
            all_encoder_hidden.append(encoder_hidden)  # 每个视图使用相同的encoder输出
        
        # 合并所有视图：[B*n_views, ...]
        decoder_input_ids = torch.cat(all_masked_input_ids, dim=0)  # [B*n_views, n_digit]
        decoder_labels = torch.cat(all_labels, dim=0)  # [B*n_views, n_digit]
        mask_positions = torch.cat(all_mask_positions, dim=0)  # [B*n_views, n_digit]
        encoder_hidden = torch.cat(all_encoder_hidden, dim=0)  # [B*n_views, seq_len*n_digit, emb_dim]
        
        # 更新batch大小并验证形状
        B_expanded = B * self.augment_factor
        
        # 形状验证
        assert decoder_input_ids.shape[0] == B_expanded, f"decoder_input_ids shape mismatch: {decoder_input_ids.shape[0]} vs {B_expanded}"
        assert decoder_labels.shape[0] == B_expanded, f"decoder_labels shape mismatch: {decoder_labels.shape[0]} vs {B_expanded}"
        assert mask_positions.shape[0] == B_expanded, f"mask_positions shape mismatch: {mask_positions.shape[0]} vs {B_expanded}"
        assert encoder_hidden.shape[0] == B_expanded, f"encoder_hidden shape mismatch: {encoder_hidden.shape[0]} vs {B_expanded}"
        
        # --- Decoder (训练模式) ---
        # 构建decoder输入嵌入
        decoder_emb = torch.zeros(B_expanded, self.n_digit, self.n_embd, device=device)
        
        for d in range(self.n_digit):
            # 获取当前digit的codebook IDs
            codebook_ids = decoder_input_ids[:, d]  # [B_expanded]
            
            # 转换为token IDs，添加安全检查
            token_ids = codebook_ids + self.tokenizer.sid_offset + d * self.codebook_size
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
            
            # 安全的embedding查找
            token_emb = self.embedding(token_ids)  # [B_expanded, emb_dim]
            
            # 获取当前digit的mask embedding
            mask_emb = self.mask_emb_table(torch.tensor(d, device=device))  # [emb_dim]
            mask_emb = mask_emb.unsqueeze(0).expand(B_expanded, -1)  # [B_expanded, emb_dim]
            
            # 根据mask_positions决定使用哪种embedding
            is_masked = mask_positions[:, d].unsqueeze(-1)  # [B_expanded, 1]
            decoder_emb[:, d, :] = torch.where(is_masked.bool(), mask_emb, token_emb)
        
        # 移除位置编码：decoder只使用掩码，不需要位置编码
        decoder_emb = self.drop(decoder_emb)
        
        # Pass through decoder blocks
        decoder_hidden = decoder_emb
        for block in self.decoder_blocks:
            block_output = block(decoder_hidden, encoder_hidden=encoder_hidden)
            decoder_hidden = block_output['hidden_states']
        
        decoder_hidden = self.ln_f(decoder_hidden)  # [B_expanded, n_digit, emb_dim]
        
        # 计算损失：只在被掩码的位置计算
        # 使用批量方式计算所有 digit 的 logits
        all_logits = self._compute_all_digit_logits(decoder_hidden)  # [B_expanded, n_digit, codebook_size]
        
        total_loss = 0.0
        total_weight = 0.0
        
        for d in range(self.n_digit):
            # 获取当前digit的logits
            logits_d = all_logits[:, d, :]  # [B_expanded, codebook_size]
            
            # 获取标签和掩码
            labels_d = decoder_labels[:, d]  # [B_expanded]
            mask_d = mask_positions[:, d].float()  # [B_expanded]
            
            # 计算交叉熵损失
            loss_d = F.cross_entropy(logits_d, labels_d, reduction='none')  # [B_expanded]
            
            # 只在被掩码的位置计算损失
            weighted_loss = loss_d * mask_d
            total_loss += weighted_loss.sum()
            total_weight += mask_d.sum()
        
        # 平均损失
        if total_weight > 0:
            total_loss = total_loss / total_weight
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        output = ModelOutput()
        output.loss = total_loss
        output.hidden_states = decoder_hidden
        output.logits = None  # 不返回所有logits，节省内存
        
        return output

    def forward_decoder_only(self, batch: dict, return_loss=False, digit=None) -> ModelOutput:
        """
        仅运行decoder部分，用于推理时的迭代预测
        
        Args:
            batch: 包含以下字段的字典：
                - decoder_input_ids: decoder输入 [B, n_digit]
                - encoder_hidden: encoder输出 [B, seq_len, emb_dim]
                - mask_positions: 掩码位置 [B, n_digit] (可选)
            digit: 要预测的digit位置
        """
        device = next(self.parameters()).device
        
        decoder_input_ids = batch['decoder_input_ids'].to(device)  # [B, n_digit]
        encoder_hidden = batch['encoder_hidden'].to(device)  # [B, seq_len, emb_dim]
        B, n_digit = decoder_input_ids.shape
        
        # 获取掩码位置，如果没有提供则假设所有位置都不被掩码
        if 'mask_positions' in batch:
            mask_positions = batch['mask_positions'].to(device)  # [B, n_digit]
        else:
            mask_positions = torch.zeros(B, n_digit, device=device)
        
        # 构建decoder输入嵌入
        decoder_emb = torch.zeros(B, n_digit, self.n_embd, device=device)
        
        for d in range(n_digit):
            # 获取当前digit的token IDs，添加安全检查
            token_ids = decoder_input_ids[:, d] + self.tokenizer.sid_offset + d * self.codebook_size
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
            token_emb = self.embedding(token_ids)  # [B, emb_dim]
            
            # 获取当前digit的mask embedding
            mask_emb = self.mask_emb_table(torch.tensor(d, device=device))  # [emb_dim]
            mask_emb = mask_emb.unsqueeze(0).expand(B, -1)  # [B, emb_dim]
            
            # 根据mask_positions决定使用哪种embedding
            is_masked = mask_positions[:, d].unsqueeze(-1)  # [B, 1]
            decoder_emb[:, d, :] = torch.where(is_masked.bool(), mask_emb, token_emb)
        
        # 移除位置编码：decoder只使用掩码，不需要位置编码
        decoder_emb = self.drop(decoder_emb)
        
        # Pass through decoder blocks
        decoder_hidden = decoder_emb
        for block in self.decoder_blocks:
            block_output = block(decoder_hidden, encoder_hidden=encoder_hidden)
            decoder_hidden = block_output['hidden_states']
        
        decoder_hidden = self.ln_f(decoder_hidden)  # [B, n_digit, emb_dim]
        
        # 计算指定digit的logits
        if digit is not None:
            logits = self._compute_digit_logits(decoder_hidden[:, digit, :], digit=digit)
            # 为了保持与 beam.py 的兼容性，在序列维度上添加一个维度
            logits = logits.unsqueeze(1)  # [B, 1, codebook_size]
        else:
            # 如果没有指定digit，计算所有位置的logits
            logits = self._compute_all_digit_logits(decoder_hidden)  # [B, n_digit, codebook_size]
        
        output = ModelOutput()
        output.hidden_states = decoder_hidden
        output.logits = logits
        
        return output

    def generate(self, batch, n_return_sequences=1):
        """
        使用迭代式掩码填充进行推理生成
        
        Args:
            batch: 包含encoder输入的批次数据
            n_return_sequences: 返回序列数量
        
        Returns:
            generated_sequences: [B, n_return_sequences, n_digit]
        """
        from .beam import iterative_mask_decode
        
        # 获取encoder输出
        with torch.no_grad():
            encoder_outputs = self.forward(batch, return_loss=False)
            encoder_hidden = encoder_outputs.hidden_states
            
            # 使用迭代式掩码解码
            generated_sequences = iterative_mask_decode(
                model=self,
                encoder_hidden=encoder_hidden,
                n_return_sequences=n_return_sequences,
                tokenizer=self.tokenizer
            )
        
        return generated_sequences

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight) 