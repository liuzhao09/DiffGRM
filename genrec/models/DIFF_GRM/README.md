# DIFF_GRM: Diffusion-based Generative Recommendation Model

## 🎯 模型概述

DIFF_GRM是一个基于Diffusion思想的生成式推荐模型，采用Encoder-Decoder架构，通过掩码预测的方式进行训练和推理。

## 💡 核心设计理念

### 1. 从自回归到掩码预测

**传统自回归模型（如RPG_ED）**：
- 训练：`[BOS, sid0, sid1, sid2]` → `[sid0, sid1, sid2, sid3]`
- 推理：逐步生成 `sid0 → sid1 → sid2 → sid3`
- 问题：训练与推理存在exposure bias

**DIFF_GRM掩码预测**：
- 训练：随机掩码部分位置，预测被掩码的token
- 推理：从全掩码开始，迭代填充各个位置
- 优势：训练与推理更一致，避免error accumulation

### 2. 数据增强策略

每个训练样本生成多个掩码版本：
```python
# 原始样本: [sid0, sid1, sid2, sid3]
# 增强样本1: [MASK, sid1, MASK, sid3]  → 预测位置0,2
# 增强样本2: [sid0, MASK, MASK, MASK] → 预测位置1,2,3  
# 增强样本3: [MASK, MASK, sid2, sid3]  → 预测位置0,1
# 增强样本4: [MASK, MASK, MASK, MASK] → 预测所有位置
```

### 3. 迭代式推理

参考TensorFlow工程代码的大规模beam search：
```python
# Step 0: 全掩码预测，选择top-256候选
[MASK, MASK, MASK, MASK] → 预测所有位置

# Step 1: 基于已填充位置，继续预测
[sid0, MASK, MASK, MASK] → top-512候选

# Step 2: 进一步填充
[sid0, sid1, MASK, MASK] → top-512候选

# Step 3: 最终填充
[sid0, sid1, sid2, sid3] → 完整序列
```

## 🏗️ 架构设计

### 1. 核心组件

- **Encoder**: 处理用户历史序列，生成context表示
- **Decoder**: 基于context和部分填充的序列，预测被掩码的位置
- **Mask Embedding**: 特殊的掩码位置表示
- **Codebook Heads**: 每个SID位置独立的分类头

### 2. 关键特性

- **无因果掩码**: Decoder不使用look-ahead mask，允许双向attention
- **位置独立**: 每个SID位置有独立的输出层
- **掩码感知**: 通过mask embedding区分已知和未知位置

### 3. 损失函数

只在被掩码的位置计算损失：
```python
total_loss = 0
for digit in range(n_digit):
    if mask_positions[digit]:  # 只有被掩码的位置
        loss += cross_entropy(logits[digit], labels[digit])
```

## 📊 配置参数

### Diffusion相关
- `mask_prob: 0.5` - 训练时的掩码概率
- `augment_factor: 4` - 数据增强倍数
- `max_history_len: 50` - 历史序列长度

### 模型架构
- `n_digit: 4` - SID长度
- `codebook_size: 256` - 每个位置的词表大小
- `encoder_n_layer: 4` - Encoder层数
- `decoder_n_layer: 4` - Decoder层数

## 🚀 使用方法

### 1. 训练命令
```bash
python main.py --model=DIFF_GRM --category=Sports_and_Outdoors \
    --lr=0.003 --temperature=0.03 --n_codebook=4 --epochs=50
```

### 2. 关键特性
- **自动数据增强**: 每个样本自动生成多个掩码版本
- **高效推理**: 使用向量化的迭代beam search
- **灵活掩码**: 支持不同的掩码策略和概率

## 🔍 与RPG_ED的对比

| 特性 | RPG_ED | DIFF_GRM |
|------|--------|----------|
| 训练方式 | 自回归Teacher Forcing | 随机掩码预测 |
| 推理方式 | 逐步生成 | 迭代填充 |
| 数据增强 | 无 | 自动多样本增强 |
| 训练难度 | Exposure Bias | 训练推理一致 |
| 推理效率 | 序列依赖 | 并行友好 |

## 📈 预期优势

1. **更好的泛化**: 掩码训练提供更丰富的学习信号
2. **训练稳定**: 避免自回归的累积误差
3. **推理灵活**: 支持部分生成和条件生成
4. **数据高效**: 自动数据增强提升样本利用率

## 🔧 技术细节

### 1. Tokenizer改进
- 支持掩码数据增强
- 自动生成多样本训练数据
- 保持与原始tokenizer的兼容性

### 2. 模型改进  
- 去除decoder的因果掩码
- 添加mask embedding table
- 独立的codebook分类头

### 3. 推理优化
- 大规模向量化beam search
- 多步骤候选筛选策略
- 高效的掩码状态管理

这种设计充分借鉴了BERT等掩码语言模型的成功经验，并结合推荐系统的特点，有望在推荐质量和训练效率上带来显著提升。 