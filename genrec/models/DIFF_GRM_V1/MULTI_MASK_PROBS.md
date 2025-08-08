# DIFF_GRM 多概率掩码数据增强

## 🎯 概述

参考TensorFlow工程代码的多视图训练策略，DIFF_GRM现在支持使用不同掩码概率进行数据增强，每个训练样本会生成多个具有不同掩码率的视图，提升模型的学习效果。

## 🔧 配置方式

### 方式1：新配置格式（推荐）

```yaml
# config.yaml
mask_probs: "1.0,0.75,0.5,0.25"    # 4种不同的掩码概率，对应4个视图
```

或者使用列表格式：

```yaml
# config.yaml  
mask_probs: [1.0, 0.75, 0.5, 0.25]    # 列表格式
```

### 方式2：兼容旧配置格式

```yaml
# config.yaml
mask_prob: 0.5          # 单一掩码概率
augment_factor: 4       # 数据增强倍数
```

**注意**：新配置 `mask_probs` 的优先级更高，如果同时存在两种配置，会使用 `mask_probs`。

## 📊 工作原理

### 1. 多概率掩码生成

每个训练样本会根据配置的掩码概率生成多个视图：

```python
# 原始SID序列: [sid0, sid1, sid2, sid3]

# 视图1 (mask_prob=1.0): 全掩码
[MASK, MASK, MASK, MASK] → 预测所有位置

# 视图2 (mask_prob=0.75): 高掩码率  
[MASK, sid1, MASK, MASK] → 预测位置0,2,3

# 视图3 (mask_prob=0.5): 中等掩码率
[MASK, sid1, sid2, MASK] → 预测位置0,3

# 视图4 (mask_prob=0.25): 低掩码率
[sid0, sid1, sid2, MASK] → 预测位置3
```

### 2. 数据增强流程

```python
def encode_decoder_input_diffusion(self, target_item, augment_sample=True):
    for i in range(self.augment_factor):
        # 根据视图索引选择对应的掩码概率
        current_mask_prob = self.mask_probs[i]
        
        # 使用当前掩码概率生成掩码模式
        mask_positions = np.random.rand(self.n_digit) < current_mask_prob
        
        # 生成对应的训练样本
        # ...
```

### 3. 训练数据扩展

- **原始训练集大小**: N个样本
- **增强后训练集大小**: N × K个样本（K为视图数量）
- **每个样本的视图**: 具有不同掩码概率的K个版本

## 🚀 使用示例

### 1. 基本配置

```bash
# 使用多概率掩码训练
python main.py --model=DIFF_GRM --dataset=AmazonReviews2014 \
    --mask_probs="1.0,0.75,0.5,0.25" \
    --lr=0.0003 --epochs=50
```

### 2. 自定义掩码概率

```yaml
# 3视图配置
mask_probs: "0.8,0.5,0.2"

# 5视图配置  
mask_probs: "1.0,0.8,0.6,0.4,0.2"

# 渐进式掩码
mask_probs: "0.9,0.7,0.5,0.3,0.1"
```

### 3. 调试信息

训练时会输出每个样本的掩码信息：

```python
# tokenizer输出包含调试字段
{
    'decoder_input_ids': [...],
    'decoder_labels': [...], 
    'mask_positions': [...],
    'mask_prob': 0.75,      # 当前样本使用的掩码概率
    'view_index': 1         # 视图索引
}
```

## 📈 预期效果

### 1. 多样化训练信号

- **高掩码率视图**: 学习全局重建能力
- **中掩码率视图**: 学习部分信息推理
- **低掩码率视图**: 学习精细化预测

### 2. 训练稳定性

- 避免单一掩码概率的局限性
- 提供更丰富的学习样本
- 提升模型泛化能力

### 3. 与TensorFlow代码对齐

- 完全兼容TensorFlow工程代码的多视图策略
- 支持相同的掩码概率配置格式
- 保持一致的训练行为

## 🔍 技术细节

### 1. 掩码概率解析

```python
# 字符串格式解析
mask_probs = [float(p) for p in "1.0,0.75,0.5,0.25".split(',')]

# 列表格式直接使用
mask_probs = [1.0, 0.75, 0.5, 0.25]
```

### 2. 视图生成策略

```python
for i in range(n_augment):
    # 按顺序使用不同的掩码概率
    current_mask_prob = self.mask_probs[i % len(self.mask_probs)]
    
    # 生成掩码模式
    mask_positions = np.random.rand(self.n_digit) < current_mask_prob
    
    # 确保至少有一个位置被掩码
    if not mask_positions.any():
        mask_positions[0] = True
```

### 3. 兼容性保证

- 旧配置自动转换为新格式
- 保持原有接口不变
- 支持渐进式迁移

## 🧪 测试验证

运行测试脚本验证功能：

```bash
cd RPG_KDD2025-main
python test_multi_mask_probs.py
```

测试内容包括：
- 字符串格式配置解析
- 列表格式配置解析  
- 多视图样本生成
- 掩码概率分布验证
- 兼容性测试

## 📝 最佳实践

### 1. 掩码概率选择

```yaml
# 推荐配置：覆盖不同难度级别
mask_probs: "1.0,0.75,0.5,0.25"

# 高难度训练：更多高掩码率视图
mask_probs: "1.0,0.9,0.8,0.6"

# 渐进式训练：平滑过渡
mask_probs: "0.8,0.6,0.4,0.2"
```

### 2. 视图数量选择

- **4视图**: 平衡效果与计算成本（推荐）
- **3视图**: 快速训练，适合小数据集
- **5-6视图**: 大数据集，追求最佳效果

### 3. 训练策略

- 初期使用较多高掩码率视图，学习全局模式
- 后期增加低掩码率视图，精细化调优
- 根据验证集表现动态调整掩码概率

这种多概率掩码策略充分借鉴了TensorFlow工程代码的成功经验，为DIFF_GRM提供了更强大和灵活的数据增强能力。 