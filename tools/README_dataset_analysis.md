# Dataset Analysis Tool

这个工具用于分析DIFF_GRM预处理数据的历史序列长度分布，帮助理解数据集的特性。

## 功能特性

- ✅ 分析训练、验证、测试三个split的数据分布
- ✅ 统计历史序列长度分布（0, 1, 2, ...）
- ✅ 计算空历史样本比例
- ✅ 输出详细的统计信息和可视化
- ✅ 支持保存分析结果到JSON/CSV文件
- ✅ 生成分布图表（可选）

## 使用方法

### 基本用法

```bash
# 分析默认category的数据
python tools/analyze_dataset.py \
    --config genrec/models/DIFF_GRM/config.yaml

# 分析指定category的数据
python tools/analyze_dataset.py \
    --config genrec/models/DIFF_GRM/config.yaml \
    --category Toys_and_Games

# 保存结果到JSON文件
python tools/analyze_dataset.py \
    --config genrec/models/DIFF_GRM/config.yaml \
    --category Sports_and_Outdoors \
    --save_json results.json

# 生成分布图
python tools/analyze_dataset.py \
    --config genrec/models/DIFF_GRM/config.yaml \
    --category Sports_and_Outdoors \
    --plot_distribution
```

### 完整参数说明

```bash
python tools/analyze_dataset.py \
    --config genrec/models/DIFF_GRM/config.yaml \  # 配置文件路径
    --category Sports_and_Outdoors \                # 数据集category
    --pad_token 0 \                                # PAD token ID
    --save_json results.json \                     # 保存JSON结果
    --save_csv results.csv \                       # 保存CSV结果
    --output_dir ./output \                        # 输出目录
    --max_samples 1000 \                          # 限制分析样本数
    --plot_distribution                            # 生成分布图
```

## 输出示例

### 控制台输出

```
============================================================
TRAIN SPLIT STATISTICS
============================================================
Total samples        : 50,000
Empty history        : 2,500 (5.00%)
Non-empty history    : 47,500 (95.00%)
Mean length          : 3.45
Median length        : 3.00
Min length           : 0
Max length           : 10

Percentiles:
  25th : 2.00
  50th : 3.00
  75th : 5.00
  90th : 7.00
  95th : 8.00
  99th : 9.00

Length distribution:
  Length  0:  2,500 ( 5.0%) ██
  Length  1:  8,750 (17.5%) ████████
  Length  2: 12,500 (25.0%) █████████████
  Length  3: 10,000 (20.0%) ██████████
  Length  4:  8,000 (16.0%) ████████
  Length  5:  5,000 (10.0%) █████
  Length  6:  2,500 ( 5.0%) ██
  Length  7:    750 ( 1.5%) █
```

### JSON输出格式

```json
{
  "config": {
    "dataset": "AmazonReviews2014",
    "category": "Sports_and_Outdoors",
    "model": "DIFF_GRM"
  },
  "train": {
    "split": "train",
    "total": 50000,
    "empty": 2500,
    "mean": 3.45,
    "median": 3.0,
    "min": 0,
    "max": 10,
    "hist": {
      "0": 2500,
      "1": 8750,
      "2": 12500,
      "3": 10000,
      "4": 8000,
      "5": 5000,
      "6": 2500,
      "7": 750
    },
    "percentiles": {
      "25th": 2.0,
      "50th": 3.0,
      "75th": 5.0,
      "90th": 7.0,
      "95th": 8.0,
      "99th": 9.0
    }
  },
  "val": { ... },
  "test": { ... }
}
```

### CSV输出格式

```csv
split,length,count,percentage
train,0,2500,5.00
train,1,8750,17.50
train,2,12500,25.00
train,3,10000,20.00
train,4,8000,16.00
train,5,5000,10.00
train,6,2500,5.00
train,7,750,1.50
val,0,250,5.00
...
```

## 可视化图表

如果使用 `--plot_distribution` 参数，会生成包含4个子图的综合图表：

1. **长度分布柱状图**: 显示各split的长度分布
2. **空历史比例对比**: 比较各split的空历史样本比例
3. **统计指标对比**: 比较均值、中位数、最小值、最大值
4. **累积分布**: 显示累积百分比分布

## 常见用途

### 1. 数据质量检查
```bash
# 检查空历史样本比例
python tools/analyze_dataset.py \
    --config genrec/models/DIFF_GRM/config.yaml \
    --category Sports_and_Outdoors
```

### 2. 数据集对比
```bash
# 对比不同category的数据分布
for category in Sports_and_Outdoors Electronics Books; do
    python tools/analyze_dataset.py \
        --config genrec/models/DIFF_GRM/config.yaml \
        --category $category \
        --save_json ${category}_analysis.json
done
```

### 3. 快速测试
```bash
# 限制样本数量进行快速分析
python tools/analyze_dataset.py \
    --config genrec/models/DIFF_GRM/config.yaml \
    --max_samples 1000
```

### 4. 生成报告
```bash
# 生成完整的分析报告
python tools/analyze_dataset.py \
    --config genrec/models/DIFF_GRM/config.yaml \
    --category Sports_and_Outdoors \
    --save_json analysis.json \
    --save_csv distribution.csv \
    --plot_distribution
```

## 注意事项

1. **配置文件**: 必须使用与训练时相同的配置文件，确保数据处理一致
2. **内存使用**: 大数据集可能需要较多内存，可以使用 `--max_samples` 限制
3. **可视化依赖**: 使用 `--plot_distribution` 需要安装 `matplotlib` 和 `seaborn`
4. **PAD token**: 如果使用非0的PAD token，请通过 `--pad_token` 参数指定

## 故障排除

### 常见错误

1. **ImportError**: 确保在项目根目录运行，或正确设置PYTHONPATH
2. **FileNotFoundError**: 检查配置文件路径是否正确
3. **KeyError**: 确保数据集中包含 `history_sid` 字段
4. **MemoryError**: 使用 `--max_samples` 限制样本数量

### 调试模式

如果遇到问题，可以添加调试信息：

```python
# 在脚本中添加调试输出
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

这个工具可以轻松扩展以支持：

- 其他数据集的统计
- 更复杂的分布分析
- 自定义可视化
- 批量分析多个数据集
- 与训练流程集成

如有需要，可以参考代码中的函数结构进行扩展。 