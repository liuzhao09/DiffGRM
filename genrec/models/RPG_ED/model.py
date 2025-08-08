# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, emb_dim, n_head, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        assert emb_dim % n_head == 0
        
        self.emb_dim = emb_dim
        self.n_head = n_head
        self.head_dim = emb_dim // n_head
        
        # 线性变换层
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        
        # Dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)
        
        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x, attention_mask=None, key_value=None, past_key_value=None, use_cache=False, is_decoder_self_attn=False):
        batch_size, seq_len, _ = x.shape
        
        # 线性变换并重塑为多头
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # 如果有key_value（cross-attention）
        if key_value is not None:
            k = key_value[0]
            v = key_value[1]
        
        # 如果有past_key_value（KV cache）
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # 自注意力掩码
                mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                mask = mask.expand(batch_size, self.n_head, seq_len, k.size(2))
                
                # 创建因果掩码（仅对decoder自注意力）
                if is_decoder_self_attn:
                    causal_mask = torch.triu(torch.ones(seq_len, k.size(2), device=x.device), diagonal=1).bool()
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_head, -1, -1)
                    mask = mask | causal_mask
                
                attn_scores = attn_scores.masked_fill(mask, float('-inf'))
            else:
                # cross-attention掩码
                attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # 应用softmax和dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.emb_dim)
        
        # 输出投影和残差dropout
        output = self.out_proj(attn_output)
        output = self.resid_drop(output)
        
        # 返回KV cache
        if use_cache:
            return output, (k, v)
        else:
            return output


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, emb_dim, n_inner, resid_drop=0.1, act='gelu'):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, n_inner)
        self.fc2 = nn.Linear(n_inner, emb_dim)
        self.resid_drop = nn.Dropout(resid_drop)
        
        if act == 'gelu':
            self.act = nn.GELU()
        elif act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()
    
    def forward(self, x):
        return self.resid_drop(self.fc2(self.act(self.fc1(x))))


class EncoderBlock(nn.Module):
    """Encoder块"""
    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1, 
                 act='gelu', layer_norm_epsilon=1e-5):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        
        self.ln2 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.ffn = FeedForward(emb_dim, n_inner, resid_drop, act)
    
    def forward(self, x, attention_mask=None):
        # 自注意力 + 残差连接（非decoder自注意力）
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask, is_decoder_self_attn=False)
        # 前馈网络 + 残差连接
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """Decoder块，包含自注意力和交叉注意力"""
    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1, 
                 act='gelu', layer_norm_epsilon=1e-5):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.self_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        
        self.ln2 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.cross_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        
        self.ln3 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.ffn = FeedForward(emb_dim, n_inner, resid_drop, act)
    
    def forward(self, x, encoder_hidden=None, attention_mask=None, 
                past_key_value=None, use_cache=False, cross_key_value=None):
        # 自注意力（decoder自注意力，需要因果掩码）
        if past_key_value is not None:
            self_attn_past = past_key_value[:2]
        else:
            self_attn_past = None
            
        self_attn_output = self.self_attn(
            self.ln1(x), 
            attention_mask=attention_mask,
            past_key_value=self_attn_past,
            use_cache=use_cache,
            is_decoder_self_attn=True  # 标记为decoder自注意力
        )
        
        if use_cache:
            out, (k, v) = self_attn_output
            x = x + out
            present_key_value = (k, v)
        else:
            x = x + self_attn_output
        
        # 交叉注意力
        if cross_key_value is not None:
            # 使用缓存的cross-attention KV
            k, v = cross_key_value
            cross_attn_output = self.cross_attn(
                self.ln2(x),
                key_value=(k, v),
                attention_mask=None
            )
            x = x + cross_attn_output
        elif encoder_hidden is not None:
            # 兼容旧接口：如果没传cache，就现算
            batch_size, seq_len, hidden_dim = encoder_hidden.shape
            n_head = self.self_attn.n_head
            d_head = hidden_dim // n_head
            k = encoder_hidden.view(batch_size, seq_len, n_head, d_head).transpose(1, 2)
            v = encoder_hidden.view(batch_size, seq_len, n_head, d_head).transpose(1, 2)
            
            cross_attn_output = self.cross_attn(
                self.ln2(x),
                key_value=(k, v),
                attention_mask=None
            )
            x = x + cross_attn_output
        
        # 前馈网络
        x = x + self.ffn(self.ln3(x))
        
        if use_cache:
            return x, present_key_value
        else:
            return x


class ModelOutput:
    """模型输出类"""
    def __init__(self):
        self.logits: Optional[torch.Tensor] = None
        self.loss: Optional[torch.Tensor] = None
        self.encoder_hidden: Optional[torch.Tensor] = None


class RPG_ED(AbstractModel):
    """RPG Encoder-Decoder模型"""
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(RPG_ED, self).__init__(config, dataset, tokenizer)

        # 常量
        self.vocab_size = tokenizer.vocab_size
        self.n_codebook = 4
        self.codebook_size = 256
        
        # 共享token嵌入（用于输入）
        self.tok_emb = nn.Embedding(self.vocab_size, config['n_embd'])
        
        # MLP层：将4个SID token压缩为1个token
        self.item_mlp = nn.Sequential(
            nn.Linear(4 * config['n_embd'], config['n_embd']),  # 4×d → d
            nn.ReLU(),
            nn.Linear(config['n_embd'], config['n_embd'])
        )
        
        # 4个独立的codebook，每个256个token
        self.codebook = nn.Parameter(
            torch.randn(self.n_codebook, self.codebook_size, config['n_embd']) * 0.02
        )
        
        # 输出投影层
        self.out_proj = nn.Linear(config['n_embd'], config['n_embd'], bias=False)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                emb_dim=config['n_embd'],
                n_head=config['n_head'],
                n_inner=config['n_inner'],
                attn_drop=config['dropout'],
                resid_drop=config['dropout'],
                act=config.get('activation_function', 'gelu'),
                layer_norm_epsilon=config.get('layer_norm_epsilon', 1e-5)
            )
            for _ in range(config['encoder_n_layer'])
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                emb_dim=config['n_embd'],
                n_head=config['n_head'],
                n_inner=config['n_inner'],
                attn_drop=config['dropout'],
                resid_drop=config['dropout'],
                act=config.get('activation_function', 'gelu'),
                layer_norm_epsilon=config.get('layer_norm_epsilon', 1e-5)
            )
            for _ in range(config['decoder_n_layer'])
        ])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(config['n_embd'], eps=config.get('layer_norm_epsilon', 1e-5))
        
        # 位置编码：为encoder添加绝对位置编码
        self.max_history_len = 50  # history_sid固定50个位置
        self.pos_emb_enc = nn.Embedding(
            self.max_history_len, config['n_embd']
        )
        
        # 语言模型头（如果共享嵌入）
        if config.get('share_embeddings', True):
            self.lm_head = None  # 使用tok_emb.weight.T
        else:
            self.lm_head = nn.Linear(config['n_embd'], self.vocab_size, bias=False)
        
        # 损失函数
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token)
        
        # 参数初始化
        self.apply(self._init_weights)
    
    def _compute_digit_logits(self, hidden_last, digit=None):
        """
        计算每个digit位置的logits
        
        Args:
            hidden_last: [B, seq_len, d] decoder的向量
            digit: 当前是第几个digit (0-3)，None表示训练模式（处理所有digit）
        
        Returns:
            logits: [B, seq_len, 256] 或 [B, 1, 256] 每个digit在对应codebook上的logits
        """
        # 输出投影
        hidden_last = self.out_proj(hidden_last)  # [B, seq_len, d]
        
        # 现在统一使用指定digit的逻辑（训练和推理一致）
        if digit is None:
            raise ValueError("digit参数不能为None，必须指定使用哪个codebook")
        
        if hidden_last.size(1) != 1:
            raise ValueError(f"Expected seq_len=1, got {hidden_last.size(1)}")
        
        cb = self.codebook[digit]  # [256,d]
        logits = torch.matmul(hidden_last.squeeze(1), cb.t())  # [B,256]
        return logits.unsqueeze(1)  # [B,1,256]

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.tok_emb.parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict, return_loss=True) -> ModelOutput:
        # --- Encoder ---
        # 获取历史SID
        history_sid = batch['history_sid']  # [B, 50, 4]
        batch_size, seq_len, n_digit = history_sid.shape
        
        # 获取token嵌入
        tok_emb = self.tok_emb(history_sid)  # [B, S, 4, d]
        B, S, _, d = tok_emb.shape
        item_emb = tok_emb.reshape(B, S, 4 * d)  # [B, S, 4·d]
        item_emb = self.item_mlp(item_emb)  # [B, S, d]
        
        # 添加位置编码
        pos_ids = torch.arange(S, device=item_emb.device)  # (S,)
        pos_emb = self.pos_emb_enc(pos_ids)  # (S, d)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, S, d)
        
        # 将位置编码加到item_emb上
        enc_hidden = item_emb + pos_emb
        
        # 创建attention mask
        enc_mask = (history_sid[:, :, 0] != self.tokenizer.pad_token).long()  # 基于第一个SID token判断
        
        # 通过encoder
        for block in self.encoder_blocks:
            enc_hidden = block(enc_hidden, enc_mask)
        
        # --- Decoder ---
        if return_loss:
            decoder_input_ids = batch['decoder_input_ids']  # [B, 4] - [BOS, sid0, sid1, sid2]
            decoder_labels = batch['decoder_labels']  # [B, 4] - [sid0, sid1, sid2, sid3]
            
            # 获取decoder输入嵌入
            dec_input = self.tok_emb(decoder_input_ids)  # [B, 4, d]
            
            # 创建 attention mask：[B, seq_len] 格式，全1表示所有位置都是有效的
            # 因果掩码由 MultiHeadAttention 内部的 is_decoder_self_attn=True 自动处理
            seq_len_dec = decoder_input_ids.size(1)
            dec_mask = torch.ones(batch_size, seq_len_dec, dtype=torch.long, device=dec_input.device)
            
            # 通过decoder
            dec_hidden = dec_input
            for block in self.decoder_blocks:
                dec_hidden = block(dec_hidden, enc_hidden, dec_mask)
            
            # 最终层归一化
            dec_hidden = self.ln_f(dec_hidden)
            
            # 正确的teacher forcing：使用所有4个位置
            # 位置0（BOS）预测sid0，位置1（sid0）预测sid1，位置2（sid1）预测sid2，位置3（sid2）预测sid3
            hidden_last = dec_hidden  # [B, 4, d] - 使用所有位置
            labels = decoder_labels  # [B, 4] - [sid0, sid1, sid2, sid3]
            
            # 分别计算每个digit的logits和loss
            total_loss = 0.0
            all_logits = []
            
            for digit in range(self.n_codebook):
                # 获取当前digit位置的hidden state
                digit_hidden = hidden_last[:, digit:digit+1, :]  # [B, 1, d]
                # 使用对应的codebook计算logits
                digit_logits = self._compute_digit_logits(digit_hidden, digit=digit)  # [B, 1, 256]
                all_logits.append(digit_logits.squeeze(1))  # [B, 256]
                
                # 计算当前digit的loss
                digit_labels = labels[:, digit]  # [B]
                digit_loss = F.cross_entropy(digit_logits.squeeze(1), digit_labels, reduction='mean')
                total_loss += digit_loss
            
            # 平均loss
            loss = total_loss / self.n_codebook
            
            # 重新组装logits为[B, 4, 256]格式（用于兼容性）
            logits = torch.stack(all_logits, dim=1)  # [B, 4, 256]
            
            outputs = ModelOutput()
            outputs.logits = logits
            outputs.loss = loss
            outputs.encoder_hidden = enc_hidden
            return outputs
        else:
            # 推理模式：这个分支通常不会被直接调用，beam search会使用forward_decoder_only
            # 但为了完整性，还是实现它
            raise NotImplementedError("推理模式请使用forward_decoder_only或generate方法")

    def forward_decoder_only(self, batch: dict, return_loss=False, digit=None) -> ModelOutput:
        """仅运行decoder（用于beam search）"""
        decoder_input_ids = batch['decoder_input_ids']
        encoder_hidden = batch['encoder_hidden']
        
        # 获取decoder输入嵌入
        dec_input = self.tok_emb(decoder_input_ids)
        
        # 获取attention mask（如果提供）
        if 'attention_mask' in batch:
            dec_mask = batch['attention_mask']
        else:
            # 创建decoder的attention mask
            batch_size, seq_len_dec = decoder_input_ids.shape
            dec_mask = torch.ones(batch_size, seq_len_dec, dtype=torch.long, device=dec_input.device)
        
        # 通过decoder
        dec_hidden = dec_input
        for block in self.decoder_blocks:
            dec_hidden = block(dec_hidden, encoder_hidden, dec_mask)
        
        # 最终层归一化
        dec_hidden = self.ln_f(dec_hidden)  # [B, L, d]
        
        # 计算logits - 必须指定digit参数
        if digit is None:
            raise ValueError("forward_decoder_only必须指定digit参数")
            
        hidden_last = dec_hidden[:, -1:, :]  # 只要最新token
        logits = self._compute_digit_logits(hidden_last, digit=digit)  # [B, 1, 256]
        
        outputs = ModelOutput()
        outputs.logits = logits
        return outputs

    def forward_decoder_with_cache(self, batch: dict, kv_cache: dict, return_loss=False, digit=None) -> ModelOutput:
        """带KV cache的decoder前向传播"""
        decoder_input_ids = batch['decoder_input_ids']
        encoder_hidden = batch['encoder_hidden']
        
        # 获取decoder输入嵌入
        dec_input = self.tok_emb(decoder_input_ids)
        
        # 获取attention mask（如果提供）
        if 'attention_mask' in batch:
            dec_mask = batch['attention_mask']
        else:
            # 创建decoder的attention mask
            batch_size, seq_len_dec = decoder_input_ids.shape
            dec_mask = torch.ones(batch_size, seq_len_dec, dtype=torch.long, device=dec_input.device)
        
        # 初始化cross-attention KV cache（第一次调用时）
        if 'self' not in kv_cache:
            kv_cache['self'] = {i: None for i in range(len(self.decoder_blocks))}
            kv_cache['cross'] = {}
            
            # 预计算所有cross-attn的K,V
            enc_h = encoder_hidden  # (B, S_enc, d)
            B, S_enc, d = enc_h.shape
            n_head = self.decoder_blocks[0].self_attn.n_head
            d_head = d // n_head
            
            # 为每层分别计算投影后的K,V
            for i, block in enumerate(self.decoder_blocks):
                proj_k = block.cross_attn.k_proj(enc_h)  # (B, S_enc, d)
                proj_v = block.cross_attn.v_proj(enc_h)  # (B, S_enc, d)
                k = proj_k.view(B, S_enc, n_head, d_head).transpose(1, 2)  # (B, h, S, d_h)
                v = proj_v.view(B, S_enc, n_head, d_head).transpose(1, 2)  # (B, h, S, d_h)
                kv_cache['cross'][i] = (k, v)
        
        # 通过decoder（使用KV cache）
        dec_hidden = dec_input
        for layer_idx, block in enumerate(self.decoder_blocks):
            past_self = kv_cache['self'].get(layer_idx)
            cross_kv = kv_cache['cross'].get(layer_idx)
            
            dec_output = block(
                dec_hidden, 
                encoder_hidden=None,  # 不再让block重算K,V
                attention_mask=dec_mask,
                past_key_value=past_self,
                use_cache=True,
                cross_key_value=cross_kv  # 传过去
            )
            
            if isinstance(dec_output, tuple):
                dec_hidden = dec_output[0]
                kv_cache['self'][layer_idx] = dec_output[1]
            else:
                dec_hidden = dec_output
        
        # 最终层归一化
        dec_hidden = self.ln_f(dec_hidden)  # [B, L, d]
        
        # 计算logits - 必须指定digit参数
        if digit is None:
            raise ValueError("forward_decoder_with_cache必须指定digit参数")
            
        hidden_last = dec_hidden[:, -1:, :]  # 只要最新token
        logits = self._compute_digit_logits(hidden_last, digit=digit)  # [B, 1, 256]
        
        outputs = ModelOutput()
        outputs.logits = logits
        return outputs

    def generate(self, batch, n_return_sequences=1):
        """生成推荐"""
        # 先运行encoder
        with torch.no_grad():
            history_sid = batch['history_sid']
            batch_size, seq_len, n_digit = history_sid.shape
            
            tok_emb = self.tok_emb(history_sid)
            B, S, _, d = tok_emb.shape
            item_emb = tok_emb.reshape(B, S, 4 * d)  # [B, S, 4·d]
            item_emb = self.item_mlp(item_emb)  # [B, S, d]
            
            # 添加位置编码
            pos_ids = torch.arange(S, device=item_emb.device)  # (S,)
            pos_emb = self.pos_emb_enc(pos_ids)  # (S, d)
            pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, S, d)
            
            # 将位置编码加到item_emb上
            enc_hidden = item_emb + pos_emb
            
            enc_input = enc_hidden
            enc_mask = (history_sid[:, :, 0] != self.tokenizer.pad_token).long()
            
            for block in self.encoder_blocks:
                enc_hidden = block(enc_hidden, enc_mask)
        
        # 使用大规模向量化beam search生成
        from genrec.models.RPG_ED.beam import beam_search_batch
        preds = beam_search_batch(
            model=self,
            encoder_hidden=enc_hidden,
            beam_size=max(512, n_return_sequences),  # 使用大规模beam search
            max_len=self.config.get('max_generation_len', 4),
            tokenizer=self.tokenizer
        )
        
        # preds: (B, beam_size, max_len) - codebook ID序列 (0-255范围)
        # 直接返回，不需要切片
        preds = preds[:, :n_return_sequences]  # (B, n_best, n_digit)
        return preds

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) 