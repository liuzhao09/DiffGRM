# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class MultiHeadAttention(nn.Module):
    """多头自注意力机制实现"""
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
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性变换并重塑为多头
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            mask = mask.expand(batch_size, self.n_head, seq_len, seq_len)
            
            # 创建因果掩码
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.n_head, -1, -1)
            
            # 组合掩码
            combined_mask = mask | causal_mask
            attn_scores = attn_scores.masked_fill(combined_mask, float('-inf'))
        
        # 应用softmax和dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.emb_dim)
        
        # 输出投影和残差dropout
        output = self.out_proj(attn_output)
        output = self.resid_drop(output)
        
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


class DecoderBlock(nn.Module):
    """解码器块，包含多头自注意力和前馈网络"""
    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1, 
                 act='gelu', layer_norm_epsilon=1e-5):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        
        self.ln2 = nn.LayerNorm(emb_dim, eps=layer_norm_epsilon)
        self.ffn = FeedForward(emb_dim, n_inner, resid_drop, act)
    
    def forward(self, x, attention_mask=None):
        # 自注意力 + 残差连接
        x = x + self.attn(self.ln1(x), attention_mask)
        # 前馈网络 + 残差连接
        x = x + self.ffn(self.ln2(x))
        return x


class LearnablePositionalEmbedding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, max_pos, emb_dim):
        super().__init__()
        self.position_embedding = nn.Embedding(max_pos, emb_dim)

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.size()
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_emb = self.position_embedding(pos_ids)
        return x + pos_emb.unsqueeze(0)


class ResBlock(nn.Module):
    """残差块模块"""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # 初始化为恒等映射
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class ModelOutput:
    """模型输出类"""
    def __init__(self):
        self.final_states: Optional[torch.Tensor] = None
        self.loss: Optional[torch.Tensor] = None


class RPG(AbstractModel):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(RPG, self).__init__(config, dataset, tokenizer)

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

        # 词嵌入层
        self.embedding = nn.Embedding(tokenizer.vocab_size, config['n_embd'])
        
        # 位置编码
        self.pos_embedding = LearnablePositionalEmbedding(tokenizer.max_token_seq_len, config['n_embd'])
        
        # 嵌入dropout
        self.emb_drop = nn.Dropout(config['embd_pdrop'])
        
        # 解码器块
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                emb_dim=config['n_embd'],
                n_head=config['n_head'],
                n_inner=config['n_inner'],
                attn_drop=config['attn_pdrop'],
                resid_drop=config['resid_pdrop'],
                act=config['activation_function'] if config['activation_function'] in ['relu', 'gelu'] else 'gelu',
                layer_norm_epsilon=config['layer_norm_epsilon']
            )
            for _ in range(config['n_layer'])
        ])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(config['n_embd'], eps=config['layer_norm_epsilon'])

        # 多头预测层
        self.n_pred_head = getattr(tokenizer, 'n_digit', 32)
        pred_head_list = []
        for i in range(self.n_pred_head):
            pred_head_list.append(ResBlock(self.config['n_embd']))
        self.pred_heads = nn.Sequential(*pred_head_list)

        # 温度参数
        self.temperature = config['temperature']
        
        # 损失函数
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=getattr(tokenizer, 'ignored_label', -100))

        # 图约束解码
        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']

        # 参数初始化
        self.apply(self._init_weights)

    def _map_item_tokens(self) -> torch.Tensor:
        """映射商品ID到语义ID"""
        item_id2tokens = torch.zeros((self.dataset.n_items, getattr(self.tokenizer, 'n_digit', 32)), dtype=torch.long)
        item2tokens = getattr(self.tokenizer, 'item2tokens', {})
        for item in item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.embedding.parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict, return_loss=True) -> ModelOutput:
        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_embs = self.embedding(input_tokens).mean(dim=-2)
        
        # 添加位置编码
        input_embs = self.pos_embedding(input_embs)
        input_embs = self.emb_drop(input_embs)
        
        # 通过解码器块
        hidden_states = input_embs
        for block in self.decoder_blocks:
            hidden_states = block(hidden_states, batch['attention_mask'])
        
        # 最终层归一化
        hidden_states = self.ln_f(hidden_states)
        
        # 多头预测
        final_states = [self.pred_heads[i](hidden_states).unsqueeze(-2) for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        
        # 创建输出对象
        outputs = ModelOutput()
        outputs.final_states = final_states
        
        if return_loss:
            assert 'labels' in batch, 'The batch must contain the labels.'
            label_mask = batch['labels'].view(-1) != -100
            selected_states = final_states.view(-1, self.n_pred_head, self.config['n_embd'])[label_mask]
            selected_states = F.normalize(selected_states, dim=-1)
            selected_states = torch.chunk(selected_states, self.n_pred_head, dim=1)
            token_emb = self.embedding.weight[1:-1]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
            token_logits = [torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / self.temperature for i in range(self.n_pred_head)]
            token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
            losses = [
                self.loss_fct(token_logits[i], token_labels[:, i] - i * self.config['codebook_size'] - 1)
                for i in range(self.n_pred_head)
            ]
            outputs.loss = torch.mean(torch.stack(losses))
        return outputs

    def build_ii_sim_mat(self):
        """构建商品-商品相似性矩阵"""
        n_items = self.dataset.n_items
        n_digit = getattr(self.tokenizer, 'n_digit', 32)
        codebook_size = getattr(self.tokenizer, 'codebook_size', 256)

        # 重塑token嵌入
        token_embs = self.embedding.weight[1:-1].view(n_digit, codebook_size, -1)

        # 归一化并计算相似性
        token_embs = F.normalize(token_embs, dim=-1)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))

        # 转换到[0,1]范围
        token_sims_01 = 0.5 * (token_sims + 1.0)

        # 准备输出相似性矩阵
        item_item_sim = torch.zeros((n_items, n_items), device=self.embedding.weight.device, dtype=torch.float32)

        # 分块填充矩阵
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)
            tokens_i = self.item_id2tokens[i_start:i_end]

            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)
                tokens_j = self.item_id2tokens[j_start:j_end]

                block_size_i = i_end - i_start
                block_size_j = j_end - j_start
                sum_block = torch.zeros((block_size_i, block_size_j), device=self.embedding.weight.device, dtype=torch.float32)

                for k in range(n_digit):
                    row_inds = tokens_i[:, k] - k * codebook_size - 1
                    col_inds = tokens_j[:, k] - k * codebook_size - 1

                    temp = token_sims_01[k].index_select(0, row_inds)
                    temp = temp.index_select(1, col_inds)
                    sum_block += temp

                avg_block = sum_block / n_digit
                item_item_sim[i_start:i_end, j_start:j_end] = avg_block

        return item_item_sim

    def build_adjacency_list(self, item_item_sim):
        """构建邻接表"""
        return torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices

    def init_graph(self):
        """初始化图"""
        self.tokenizer.log("Building item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)
        self.tokenizer.log("Graph initialized.")

    def graph_propagation(self, token_logits, n_return_sequences):
        """图传播"""
        batch_size = token_logits.shape[0]

        # 随机采样候选节点
        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items,
            (batch_size, self.config['num_beams']),
            dtype=torch.long,
            device=token_logits.device
        )

        for sid in range(self.propagation_steps):
            # 找到邻居节点
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)

            next_nodes = []
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])
                scores = torch.gather(
                    input=token_logits[batch_id].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                    dim=-1,
                    index=(self.item_id2tokens[neighbors_in_batch] - 1)
                ).mean(dim=-1)

                idxs = torch.topk(scores, self.config['num_beams']).indices
                next_nodes.append(neighbors_in_batch[idxs])
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)
        return topk_nodes_sorted[:,:n_return_sequences].unsqueeze(-1)

    def generate(self, batch, n_return_sequences=1):
        """生成推荐"""
        outputs = self.forward(batch, return_loss=False)
        if outputs.final_states is None:
            raise ValueError("final_states should not be None")
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.config['n_embd'])
        )
        states = F.normalize(states, dim=-1)

        token_emb = self.embedding.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        logits = [torch.matmul(states[:,0,i,:], token_embs[i].T) / self.temperature for i in range(self.n_pred_head)]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)

        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            outputs = self.graph_propagation(
                token_logits=token_logits,
                n_return_sequences=n_return_sequences
            )
            return outputs
        else:
            item_logits = torch.gather(
                input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),
                dim=-1,
                index=(self.item_id2tokens[1:,:] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)
            ).mean(dim=-1)
            preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
            return preds.unsqueeze(-1)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) 