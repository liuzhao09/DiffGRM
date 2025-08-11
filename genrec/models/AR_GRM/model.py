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

        # 保存拼接后的完整k和v用于cache（在reshape之前）
        k_for_cache = k
        v_for_cache = v

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

        # Prepare cache for next iteration - 保存原始的3维k和v
        present_key_value = (k_for_cache, v_for_cache) if use_cache else None

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
        # 自注意力（支持传入因果掩码）
        self_past_kv = None
        cross_past_kv = None
        if past_key_value is not None:
            if len(past_key_value) >= 1:
                self_past_kv = past_key_value[0]
            if len(past_key_value) >= 2:
                cross_past_kv = past_key_value[1]
        
        attn_output, present_key_value = self.self_attn(
            self.ln_1(x), 
            attention_mask=attention_mask,  # 允许因果掩码
            past_key_value=self_past_kv,
            use_cache=use_cache,
            is_decoder_self_attn=True
        )
        x = x + attn_output

        # 交叉注意力
        if encoder_hidden is not None:
            if cross_key_value is not None:
                # 🚀 使用预计算的KV，避免重复计算
                encoder_kv = cross_key_value
            else:
                # 兼容旧逻辑：重新计算（仅用于非优化路径）
                encoder_kv = torch.cat([encoder_hidden, encoder_hidden], dim=-1)  # Concat K and V
            
            cross_attn_output, cross_present = self.cross_attn(
                self.ln_2(x),
                key_value=encoder_kv,
                past_key_value=cross_past_kv,
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


class AR_GRM(AbstractModel):

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
        
        # 自回归特有设置
        self.use_causal_mask = bool(config.get('use_causal_mask', True))
        
        # Embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.n_embd)
        
        # 添加与RPG_ED一致的item_mlp：将n_digit个SID token压缩为1个token
        self.item_mlp = nn.Sequential(
            nn.Linear(self.n_digit * self.n_embd, self.n_embd),  # n_digit×d → d
            nn.ReLU(),
            nn.Linear(self.n_embd, self.n_embd)
        )
        # BOS embedding（用于自回归 decoder 起始）
        self.bos_embedding = nn.Parameter(torch.randn(self.n_embd) * 0.02)
        
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
        
        # -- 共享 embedding dot-product 作为输出层 --
        share_out = self.config.get('share_decoder_output_embedding', True)
        if share_out:
            # 直接 weight-tying，不新增参数
            self.output_adapter = nn.Identity()
            print(f"[AR_GRM] Using shared embedding dot-product output layer")
        else:
            # 若以后要回滚到独立 head，用这一行
            self.output_adapter = nn.Linear(self.n_embd, self.n_embd, bias=False)
            print(f"[AR_GRM] Using independent Linear output adapter")
        # -------------------------------------------------------------
        
        # Dropout
        self.drop = nn.Dropout(self.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _compute_digit_logits(self, hidden_last, digit):
        """
        使用共享embedding的dot-product计算logits
        
        Args:
            hidden_last: (B, d_model) - decoder输出的隐藏状态
            digit: 0..n_digit-1 - 要预测的digit位置
            
        Returns:
            logits: (B, codebook_size) - 预测logits
        """
        if digit is None:
            raise ValueError("digit参数不能为None，必须指定要计算的codebook位置")
        
        if digit >= self.n_digit:
            raise ValueError(f"digit={digit} 超出范围，应该在 [0, {self.n_digit-1}]")
        
        # 2.1 取出 embedding matrix 的相应切片
        # token ID 布局 = [PAD, BOS, EOS, digit0 256 个, digit1 256 个, ...]
        start = self.tokenizer.sid_offset + digit * self.codebook_size
        end = start + self.codebook_size  # 不含 end
        # shape: (codebook_size, d_model)
        E_sub = self.embedding.weight[start:end]
        
        # 2.2 optional adapter
        h = self.output_adapter(hidden_last)  # (B, d_model)
        
        # 2.3 dot-product 得 logits
        # (B, d_model) @ (d_model, codebook_size).T → (B, codebook_size)
        logits = torch.matmul(h, E_sub.t())
        
        return logits

    @property
    def n_parameters(self) -> str:
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return f"{n_params:,}"

    def _causal_mask(self, B: int, T: int, device):
        mask = torch.tril(torch.ones(T, T, device=device))
        return mask.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)

    def forward(self, batch: dict, return_loss=True) -> ModelOutput:
        """
        自回归训练：teacher forcing（输入=[BOS, sid0..sid3]，在位置0..3预测sid0..3）
        """
        device = next(self.parameters()).device
        
        if not hasattr(self, '_debug_printed'):
            print(f"[AR_GRM] Encoder: MLP compression + abs pos encoding")
            print(f"[AR_GRM] vocab_size: {self.vocab_size}, codebook_size: {self.codebook_size}")
            self._debug_printed = True
        
        # --- Encoder ---
        history_sid = batch['history_sid'].to(device)  # [B, seq_len, n_digit]
        B, seq_len, n_digit = history_sid.shape
        
        # 与RPG_ED保持一致的处理方式
        # history_sid 已经是 token id（包含了 sid_offset 与 digit 偏移），直接使用即可
        history_tokens = history_sid.long().clamp(0, self.vocab_size - 1)
        
        # 2. 获取token嵌入
        tok_emb = self.embedding(history_tokens)  # [B, seq_len, n_digit, d]
        B, S, _, d = tok_emb.shape
        
        # 3. 重塑并通过MLP压缩：n_digit个SID token → 1个item token
        item_emb = tok_emb.reshape(B, S, self.n_digit * d)  # [B, S, n_digit*d]
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
        
        encoder_hidden = self.ln_f(encoder_hidden)  # [B, S, d]

        if not return_loss:
            out = ModelOutput()
            out.hidden_states = encoder_hidden
            return out

        # Teacher forcing inputs
        dec_gt = torch.clamp(batch['decoder_input_ids'].to(device), 0, self.codebook_size - 1)  # [B, n_digit]
        labels = torch.clamp(batch['decoder_labels'].to(device), 0, self.codebook_size - 1)      # [B, n_digit]
        B = dec_gt.size(0)

        # Convert codebook id -> token id per digit
        token_ids = []
        for d in range(self.n_digit):
            tok = dec_gt[:, d] + self.tokenizer.sid_offset + d * self.codebook_size
            tok = torch.clamp(tok, 0, self.vocab_size - 1)
            token_ids.append(tok)
        token_ids = torch.stack(token_ids, dim=1)  # [B, n_digit]

        tok_emb = self.embedding(token_ids)  # [B, n_digit, d]
        bos = self.bos_embedding.unsqueeze(0).unsqueeze(1).expand(B, 1, -1)  # [B,1,d]
        dec_inp = torch.cat([bos, tok_emb], dim=1)  # [B, n_digit+1, d]
        dec_inp = self.drop(dec_inp)

        # Decoder with causal mask and cross-attn
        x = dec_inp
        attn_mask = self._causal_mask(B, x.size(1), device) if self.use_causal_mask else None
        # 预计算 cross-KV（每层一次）
        encoder_kv_list = []
        for blk in self.decoder_blocks:
            kv_proj = blk.cross_attn.qkv(encoder_hidden)
            k = kv_proj[..., self.n_embd:2*self.n_embd]
            v = kv_proj[..., 2*self.n_embd:]
            encoder_kv_list.append(torch.cat([k, v], dim=-1))

        for i, blk in enumerate(self.decoder_blocks):
            out = blk(x, encoder_hidden=encoder_hidden, attention_mask=attn_mask,
                      past_key_value=None, use_cache=False, cross_key_value=encoder_kv_list[i])
            x = out['hidden_states']

        x = self.ln_f(x)  # [B, n_digit+1, d]

        # Compute losses at positions 0..n_digit-1 predicting sid0..sid(n-1)
        total_loss = 0.0
        for d in range(self.n_digit):
            logits_d = self._compute_digit_logits(x[:, d, :], digit=d)
            total_loss = total_loss + F.cross_entropy(
                logits_d, labels[:, d], reduction='mean',
                label_smoothing=self.config.get('label_smoothing', 0.0)
            )
        total_loss = total_loss / self.n_digit

        out = ModelOutput()
        out.loss = total_loss
        out.hidden_states = x
        return out

    def _precompute_cross_kv(self, encoder_hidden):
        kv_list = []
        for blk in self.decoder_blocks:
            with torch.no_grad():
                qkv = blk.cross_attn.qkv(encoder_hidden)
                k = qkv[..., self.n_embd:2*self.n_embd]
                v = qkv[..., 2*self.n_embd:]
                kv_list.append(torch.cat([k, v], dim=-1))
        return kv_list

    def _decode_step(self, x_last, cross_kv_list, past_kv=None):
        # x_last: [N,1,d]
        x = x_last
        present = []
        for i, blk in enumerate(self.decoder_blocks):
            self_out, self_present = self.self_attend_step(blk, x, past_kv[i] if past_kv else None)
            x = x + self_out
            cross_out, cross_present = blk.cross_attn(blk.ln_2(x), key_value=cross_kv_list[i], use_cache=True)
            x = x + cross_out
            x = x + blk.mlp(blk.ln_3(x))
            present.append((self_present, cross_present))
        x = self.ln_f(x)
        return x, present

    def self_attend_step(self, blk: DecoderBlock, x, past_kv_layer=None):
        # 单步自注意，query_len=1，无需显式掩码
        out, present = blk.self_attn(blk.ln_1(x), past_key_value=past_kv_layer, use_cache=True, is_decoder_self_attn=True)
        return out, present

    def generate(self, batch, n_return_sequences=10, mode=None):
        """顺序自回归 beam search，按 digit0→digit1→... 生成。
        返回 [B, top_k_final, n_digit]
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                enc_out = self.forward(batch, return_loss=False)
                encoder_hidden = enc_out.hidden_states
                B = encoder_hidden.size(0)
                cfg = self.config.get('ar_beam_search', {})
                pre_cut = cfg.get('pre_cut_num', [256]*self.n_digit)
                beam_num = cfg.get('beam_search_num', [256]*self.n_digit)
                TOPK = min(cfg.get('top_k_final', n_return_sequences), n_return_sequences)

                # 预计算 cross-KV
                cross_kv = self._precompute_cross_kv(encoder_hidden)

                device = encoder_hidden.device
                bos = self.bos_embedding.to(device).unsqueeze(0).unsqueeze(1).expand(B, 1, -1)
                # step0
                x, present = self._decode_step(bos, cross_kv, past_kv=None)
                logits0 = self._compute_digit_logits(x[:, 0, :], digit=0)
                logp0 = F.log_softmax(logits0, dim=-1)

                topk_p0, topk_i0 = torch.topk(logp0, k=pre_cut[0], dim=-1)
                keep_k = min(beam_num[0], pre_cut[0])  # 防止步0的keep超过pre_cut
                best_lp, best_idx = torch.topk(topk_p0, k=keep_k, dim=-1)
                tok0 = topk_i0.gather(1, best_idx)

                # expand caches to beams
                def expand_to_beam(t, k):
                    return t.unsqueeze(1).repeat(1, k, *([1]*(t.ndim-1))).view(B*k, *t.shape[1:])

                beam_self_kv = []
                for l in range(len(self.decoder_blocks)):
                    self_k, cross_k = present[l]
                    if self_k is None:
                        beam_self_kv.append(None)
                    else:
                        k,v = self_k
                        beam_self_kv.append((expand_to_beam(k, keep_k), expand_to_beam(v, keep_k)))

                def emb_of_digit_token(digit, codebook_id):
                    token_id = codebook_id + self.tokenizer.sid_offset + digit * self.codebook_size
                    token_id = torch.clamp(token_id, 0, self.vocab_size - 1)
                    return self.embedding(token_id).unsqueeze(1)

                beams = tok0.view(-1, 1)   # [B*beam0, 1]
                lp = best_lp.view(-1)
                last_emb = emb_of_digit_token(0, beams[:, -1])
                cross_kv_exp = [expand_to_beam(kv, keep_k) for kv in cross_kv]

                for d in range(1, self.n_digit):
                    x, present = self._decode_step(last_emb, cross_kv_exp, past_kv=beam_self_kv)
                    logit = self._compute_digit_logits(x[:, 0, :], digit=d)
                    logp = F.log_softmax(logit, dim=-1)

                    pre = pre_cut[d]
                    # 防止 keep 超过候选的上限（父数 * pre）
                    keep = min(beam_num[d], (beams.size(0) // B) * pre)
                    tk_prob, tk_idx = torch.topk(logp, k=pre, dim=-1)  # [N,pre]
                    cand_lp = (lp.unsqueeze(1) + tk_prob).view(B, -1)
                    best_lp, best_flat = torch.topk(cand_lp, k=keep, dim=-1)
                    parent = best_flat // pre
                    token = best_flat % pre

                    N = beams.size(0) // B
                    idx = (torch.arange(B, device=device).unsqueeze(1)*N + parent).view(-1)
                    beams = torch.cat([beams[idx], tk_idx.view(B, -1, pre).gather(2, token.unsqueeze(-1)).view(-1,1)], dim=1)
                    lp = best_lp.view(-1)

                    new_self_kv = []
                    for l in range(len(self.decoder_blocks)):
                        pk = beam_self_kv[l]
                        npk = present[l][0] if present[l] is not None else None
                        if pk is None or npk is None:
                            new_self_kv.append(None)
                        else:
                            old_k, old_v = pk
                            add_k, add_v = npk
                            old_k = old_k[idx]; old_v = old_v[idx]
                            add_k = add_k[idx]; add_v = add_v[idx]
                            new_self_kv.append((torch.cat([old_k, add_k], dim=1), torch.cat([old_v, add_v], dim=1)))
                    beam_self_kv = new_self_kv
                    last_emb = emb_of_digit_token(d, beams[:, -1])

                beams = beams.view(B, -1, self.n_digit)   # [B, num_beams, n_digit]
                lp = lp.view(B, -1)                       # [B, num_beams]

                # === Legality filtering among current top beams ===
                # Sort beams by log-probability descending, then take the top-K legal sequences
                K = min(TOPK, lp.size(1))
                sorted_lp, sorted_idx = torch.sort(lp, dim=-1, descending=True)
                sorted_idx_exp = sorted_idx.unsqueeze(-1).expand(-1, -1, self.n_digit)
                sorted_beams = beams.gather(1, sorted_idx_exp)  # [B, num_beams, n_digit]

                final_list = []
                num_beams = sorted_beams.size(1)
                for b in range(B):
                    selected = []
                    # iterate over sorted candidates; keep only legal sequences
                    for j in range(num_beams):
                        seq = sorted_beams[b, j].tolist()
                        if self.tokenizer.codebooks_to_item_id(seq) is not None:
                            selected.append(sorted_beams[b, j])
                            if len(selected) >= K:
                                break
                    # Fallbacks: ensure we always return K sequences
                    if len(selected) == 0:
                        # if no legal sequence found, fall back to the best candidate
                        selected.append(sorted_beams[b, 0])
                    while len(selected) < K:
                        selected.append(selected[-1])
                    final_list.append(torch.stack(selected, dim=0))

                final = torch.stack(final_list, dim=0)  # [B, K, n_digit]
                return final
        finally:
            if was_training:
                self.train()

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
        # 注意：output_adapter如果是Identity()，不需要初始化 