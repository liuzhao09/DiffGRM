# RPG_ED: Encoder-Decoder RPG Model

## 概述

RPG_ED是一个基于Encoder-Decoder架构的推荐系统模型，将原有的Decoder-only RPG模型改造为Encoder-Decoder结构。

## 主要特性

### 1. Encoder-Decoder架构
- **Encoder**: 处理用户历史信息（用户嵌入 + 50个历史商品）
- **Decoder**: 自回归生成目标商品的SID序列
- **Cross-Attention**: Decoder通过交叉注意力机制利用Encoder的信息

### 2. 简化的SID配置
- **n_digit**: 4层（每层256个token）
- **codebook_size**: 256
- **vocab_size**: 3 + n_user_buckets + 4×256 = 11027

### 3. 用户嵌入
- 为每个用户训练独立的嵌入向量
- 使用桶化策略处理大量用户（n_user_buckets=10000）

### 4. 历史信息处理
- 将50个历史商品的4个SID token通过MLP压缩为单个token
- 维度: (50, 4, dim) → (50, dim) → MLP → (50, dim)

### 5. 生成策略
- **训练**: Next-token预测，因果掩码
- **推理**: Beam search + KV cache
- **输出**: 4个token的SID序列，通过倒排索引转换为商品ID

## 文件结构

```
genrec/models/RPG_ED/
├── config.yaml          # 模型配置文件
├── model.py            # 主模型实现
├── tokenizer.py        # Tokenizer实现
├── beam.py             # Beam search生成器
├── collate.py          # 数据批处理函数
└── README.md           # 说明文档
```

## 配置参数

```yaml
# 模型架构
model: RPG_ED
n_digit: 4
codebook_size: 256
share_embeddings: true

# Encoder/Decoder层数
encoder_n_layer: 4
decoder_n_layer: 4
n_head: 8
n_embd: 448
n_inner: 1024
dropout: 0.1
embd_pdrop: 0.1
attn_pdrop: 0.1
resid_pdrop: 0.1

# 生成参数
beam_size: 10
max_generation_len: 4

# 用户配置
n_user_buckets: 10000
```

## 使用方法

### 1. 训练

```python
from genrec.models.RPG_ED.model import RPG_ED
from genrec.models.RPG_ED.tokenizer import RPG_EDTokenizer

# 创建tokenizer
tokenizer = RPG_EDTokenizer(config, dataset)

# 创建模型
model = RPG_ED(config, dataset, tokenizer)

# 训练
trainer = Trainer(config, model, tokenizer)
trainer.fit(train_dataloader, val_dataloader)
```

### 2. 推理

```python
# 生成推荐
predictions = model.generate(batch)
# predictions: [batch_size, n_return_sequences, 1]
```

## 数据格式

### 训练数据
```python
{
    'user_ids': [batch_size],           # 用户ID
    'history_sid': [batch_size, 50, 4], # 历史SID序列
    'decoder_input_ids': [batch_size, 5], # [BOS] + 前3个SID
    'decoder_labels': [batch_size, 5]   # 完整4个SID + [EOS]
}
```

### 推理数据
```python
{
    'user_ids': [batch_size],           # 用户ID
    'history_sid': [batch_size, 50, 4], # 历史SID序列
}
```

## 核心组件

### 1. MultiHeadAttention
- 支持自注意力和交叉注意力
- 支持KV cache用于推理优化
- 支持因果掩码和padding掩码
- 区分Encoder和Decoder的注意力掩码

### 2. EncoderBlock
- 自注意力 + 前馈网络
- 残差连接 + 层归一化
- 仅使用padding掩码，无因果掩码

### 3. DecoderBlock
- 自注意力（因果掩码）
- 交叉注意力（与Encoder输出）
- 前馈网络

### 4. Beam Search
- 支持beam search生成
- 支持KV cache优化
- 支持贪心生成（调试用）

## 存储优化

### 1. 正排索引
- 文件: `item_id2tokens.npy`
- 格式: [n_items, 4] 的numpy数组
- 用途: 商品ID → SID序列

### 2. 倒排索引
- 文件: `tokens2item.pkl`
- 格式: {(sid0, sid1, sid2, sid3): item_id}
- 用途: SID序列 → 商品ID

### 3. 内存占用
- 1M商品 ≈ 8MB（4字节×4层×1M商品）
- 完全可常驻内存

## 性能优化

### 1. KV Cache
- Encoder: 一次性计算并缓存
- Decoder: 增量缓存，避免重复计算

### 2. Beam Search
- 并行处理多个候选
- 早期停止机制
- 去重和排序

### 3. 批处理
- 支持批量推理
- 动态批处理大小

## 与原始RPG的对比

| 特性 | 原始RPG | RPG_ED |
|------|---------|--------|
| 架构 | Decoder-only | Encoder-Decoder |
| SID层数 | 16 | 4 |
| 用户嵌入 | ❌ | ✅ |
| 历史处理 | 简单拼接 | MLP压缩 |
| 生成策略 | 图约束 | Beam search |
| 损失函数 | 多头损失 | Next-token CE |
| 推理优化 | 图传播 | KV cache |

## 测试

运行测试脚本验证模型功能：

```bash
python test_rpg_ed.py
```

## 注意事项

1. **SID生成**: 需要先运行OPQ生成语义ID
2. **用户桶化**: 大量用户时使用桶化策略
3. **内存管理**: 推理时注意KV cache的内存占用
4. **索引更新**: 新增商品时需要更新正排/倒排索引
5. **ID区间**: 确保用户token和SID token的ID区间不冲突

## 未来改进

1. **动态SID**: 根据商品复杂度动态调整SID层数
2. **多模态**: 支持图像、文本等多模态信息
3. **在线学习**: 支持增量学习和在线更新
4. **分布式**: 支持大规模分布式训练和推理 