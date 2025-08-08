#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset Analysis Tool for DIFF_GRM

分析DIFF_GRM预处理数据的历史序列长度分布，支持：
- 分析训练、验证、测试三个split的数据分布
- 统计历史序列长度分布（0, 1, 2, ...）
- 计算空历史样本比例
- 输出详细的统计信息和可视化
- 支持保存分析结果到JSON文件
"""

import argparse
import os
import sys
import yaml
import json
import numpy as np
from collections import Counter
from typing import Dict, Any, List
from datetime import datetime

# 添加项目根目录到path
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJ_ROOT)

# 导入项目模块
from genrec.utils import get_dataset, get_tokenizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="DIFF_GRM Dataset Analyzer")
    
    # 基本参数
    parser.add_argument("--config", required=True,
                       help="配置文件路径 (YAML格式)")
    parser.add_argument("--category", default=None,
                       help="数据集category，如果不指定则使用config中的设置")
    parser.add_argument("--pad_token", type=int, default=0,
                       help="PAD token的ID，默认为0")
    
    # 输出控制
    parser.add_argument("--save_json", default=None,
                       help="保存分析结果到JSON文件")
    parser.add_argument("--save_csv", default=None,
                       help="保存详细分布到CSV文件")
    parser.add_argument("--output_dir", default=None,
                       help="输出目录，默认与config同目录")
    
    # 分析控制
    parser.add_argument("--max_samples", type=int, default=None,
                       help="限制分析样本数量（用于快速测试）")
    parser.add_argument("--print_details", action="store_true",
                       help="打印详细的分布信息")
    parser.add_argument("--plot_distribution", action="store_true",
                       help="生成分布图（需要matplotlib）")
    
    return parser.parse_args()


def load_config(config_path: str, category: str = None) -> Dict[str, Any]:
    """加载配置文件"""
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 覆盖category
    if category:
        config['category'] = category
        print(f"Override category: {category}")
    
    # 确保必要的字段存在
    config.setdefault('dataset', 'AmazonReviews2014')
    config.setdefault('model', 'DIFF_GRM')
    config.setdefault('device', 'cpu')
    
    # 参考Pipeline类的初始化方式，设置必要的配置
    from genrec.utils import init_device, init_seed, init_logger, get_config
    from accelerate import Accelerator
    from logging import getLogger
    
    # 使用get_config函数来正确初始化配置
    # 这会加载默认配置、数据集配置、模型配置等
    final_config = get_config(
        model_name=config.get('model', 'DIFF_GRM'),
        dataset_name=config.get('dataset', 'AmazonReviews2014'),
        config_file=config_path,
        config_dict=config  # 我们的配置会覆盖默认配置
    )
    
    # 设置设备和分布式训练
    final_config['device'], final_config['use_ddp'] = init_device()
    
    # 设置accelerator（简化版本，不需要tensorboard）
    try:
        project_dir = os.path.join(
            final_config.get('tensorboard_log_dir', 'runs'),
            final_config.get('dataset', 'AmazonReviews2014'),
            final_config.get('model', 'DIFF_GRM')
        )
        accelerator = Accelerator(log_with='tensorboard', project_dir=project_dir)
    except:
        # 如果tensorboard不可用，使用无日志的accelerator
        accelerator = Accelerator()
        print("Warning: Tensorboard not available, using accelerator without logging")
    
    final_config['accelerator'] = accelerator
    
    # 初始化种子和日志
    init_seed(final_config.get('rand_seed', final_config.get('seed', 42)), 
              final_config.get('reproducibility', True))
    init_logger(final_config)
    logger = getLogger()
    
    print(f"Dataset: {final_config.get('dataset')}")
    print(f"Category: {final_config.get('category')}")
    print(f"Model: {final_config.get('model')}")
    print(f"Device: {final_config['device']}")
    
    return final_config


def calculate_history_length(example: Dict[str, Any], pad_token: int = 0) -> int:
    """
    计算单个样本的历史序列长度（非PAD的行数）
    
    Args:
        example: 包含history_sid的样本
        pad_token: PAD token的ID
    
    Returns:
        int: 历史序列长度
    """
    history_sid = example['history_sid']
    
    if isinstance(history_sid, list):
        # 如果是list格式，计算非PAD的序列数
        non_pad_count = 0
        for seq in history_sid:
            if isinstance(seq, list):
                # 检查序列是否包含非PAD token
                if any(token != pad_token for token in seq):
                    non_pad_count += 1
        return non_pad_count
    else:
        # 如果是tensor格式，转换为numpy处理
        import torch
        if torch.is_tensor(history_sid):
            history_sid = history_sid.cpu().numpy()
        
        # 计算非PAD的行数
        non_pad_rows = 0
        for seq in history_sid:
            if np.any(seq != pad_token):
                non_pad_rows += 1
        return non_pad_rows


def analyze_split(tokenized_data: List[Dict[str, Any]], 
                 split_name: str,
                 pad_token: int = 0,
                 max_samples: int = None) -> Dict[str, Any]:
    """
    分析单个split的数据分布
    
    Args:
        tokenized_data: tokenized后的数据集
        split_name: split名称
        pad_token: PAD token的ID
        max_samples: 最大分析样本数
    
    Returns:
        Dict: 包含统计信息的字典
    """
    print(f"\nAnalyzing {split_name} split...")
    
    # 限制样本数量
    if max_samples:
        tokenized_data = tokenized_data[:max_samples]
        print(f"Limited to {max_samples} samples for analysis")
    
    # 计算每个样本的历史长度
    lengths = []
    for i, example in enumerate(tokenized_data):
        try:
            length = calculate_history_length(example, pad_token)
            lengths.append(length)
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    if not lengths:
        print(f"Warning: No valid samples found in {split_name} split")
        return {
            "split": split_name,
            "total": 0,
            "empty": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0,
            "max": 0,
            "hist": {},
            "percentiles": {}
        }
    
    # 转换为numpy数组进行统计
    lengths_array = np.array(lengths)
    
    # 统计信息
    total = len(lengths_array)
    empty = np.sum(lengths_array == 0)
    mean = float(np.mean(lengths_array))
    median = float(np.median(lengths_array))
    min_len = int(np.min(lengths_array))
    max_len = int(np.max(lengths_array))
    
    # 长度分布
    counter = Counter(lengths)
    hist = dict(sorted(counter.items()))
    
    # 百分位数
    percentiles = {
        "25th": float(np.percentile(lengths_array, 25)),
        "50th": float(np.percentile(lengths_array, 50)),
        "75th": float(np.percentile(lengths_array, 75)),
        "90th": float(np.percentile(lengths_array, 90)),
        "95th": float(np.percentile(lengths_array, 95)),
        "99th": float(np.percentile(lengths_array, 99))
    }
    
    return {
        "split": split_name,
        "total": total,
        "empty": empty,
        "mean": mean,
        "median": median,
        "min": min_len,
        "max": max_len,
        "hist": hist,
        "percentiles": percentiles
    }


def print_statistics(stats: Dict[str, Any]):
    """打印统计信息"""
    split_name = stats['split']
    total = stats['total']
    empty = stats['empty']
    
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} SPLIT STATISTICS")
    print(f"{'='*60}")
    
    if total == 0:
        print("No data available")
        return
    
    print(f"Total samples        : {total:,}")
    print(f"Empty history        : {empty:,} ({empty/total*100:.2f}%)")
    print(f"Non-empty history    : {total-empty:,} ({(total-empty)/total*100:.2f}%)")
    print(f"Mean length          : {stats['mean']:.2f}")
    print(f"Median length        : {stats['median']:.2f}")
    print(f"Min length           : {stats['min']}")
    print(f"Max length           : {stats['max']}")
    
    # 百分位数
    print(f"\nPercentiles:")
    for p_name, p_value in stats['percentiles'].items():
        print(f"  {p_name:>4} : {p_value:.2f}")
    
    # 长度分布
    print(f"\nLength distribution:")
    hist = stats['hist']
    for length in sorted(hist.keys()):
        count = hist[length]
        percentage = count / total * 100
        bar = "█" * int(percentage / 2)  # 简单的文本条形图
        print(f"  Length {length:2d}: {count:6,} ({percentage:5.1f}%) {bar}")


def save_results(results: Dict[str, Any], args):
    """保存分析结果"""
    if not args.save_json and not args.save_csv:
        return
    
    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.config)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    category = results.get('config', {}).get('category', 'unknown')
    base_name = f"dataset_analysis_{category}_{timestamp}"
    
    # 保存JSON
    if args.save_json:
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {json_path}")
    
    # 保存CSV
    if args.save_csv:
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        import csv
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['split', 'length', 'count', 'percentage'])
            
            for split_name, split_stats in results.items():
                if split_name == 'config':
                    continue
                
                total = split_stats['total']
                if total == 0:
                    continue
                
                for length, count in split_stats['hist'].items():
                    percentage = count / total * 100
                    writer.writerow([split_name, length, count, f"{percentage:.2f}"])
        
        print(f"Detailed distribution saved to: {csv_path}")


def plot_distribution(results: Dict[str, Any]):
    """绘制分布图（可选）"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib or seaborn not available, skipping plot generation")
        return
    
    # 设置样式
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dataset History Length Distribution', fontsize=16)
    
    splits = [k for k in results.keys() if k != 'config']
    
    # 1. 长度分布柱状图
    ax1 = axes[0, 0]
    for split in splits:
        hist = results[split]['hist']
        lengths = list(hist.keys())
        counts = list(hist.values())
        ax1.bar(lengths, counts, alpha=0.7, label=split, width=0.8)
    
    ax1.set_xlabel('History Length')
    ax1.set_ylabel('Count')
    ax1.set_title('Length Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 空历史比例饼图
    ax2 = axes[0, 1]
    empty_counts = [results[split]['empty'] for split in splits]
    non_empty_counts = [results[split]['total'] - results[split]['empty'] for split in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    ax2.bar(x - width/2, empty_counts, width, label='Empty History', alpha=0.7)
    ax2.bar(x + width/2, non_empty_counts, width, label='Non-empty History', alpha=0.7)
    
    ax2.set_xlabel('Split')
    ax2.set_ylabel('Count')
    ax2.set_title('Empty vs Non-empty History')
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 统计指标对比
    ax3 = axes[1, 0]
    metrics = ['mean', 'median', 'min', 'max']
    x = np.arange(len(metrics))
    width = 0.8 / len(splits)
    
    for i, split in enumerate(splits):
        values = [results[split][metric] for metric in metrics]
        ax3.bar(x + i*width, values, width, label=split, alpha=0.7)
    
    ax3.set_xlabel('Metric')
    ax3.set_ylabel('Value')
    ax3.set_title('Statistical Metrics')
    ax3.set_xticks(x + width * (len(splits) - 1) / 2)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 累积分布
    ax4 = axes[1, 1]
    for split in splits:
        hist = results[split]['hist']
        total = results[split]['total']
        
        lengths = sorted(hist.keys())
        cumulative = []
        cumsum = 0
        for length in lengths:
            cumsum += hist[length]
            cumulative.append(cumsum / total * 100)
        
        ax4.plot(lengths, cumulative, marker='o', label=split, linewidth=2, markersize=4)
    
    ax4.set_xlabel('History Length')
    ax4.set_ylabel('Cumulative Percentage (%)')
    ax4.set_title('Cumulative Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    category = results.get('config', {}).get('category', 'unknown')
    plot_path = f"dataset_analysis_{category}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {plot_path}")
    
    plt.show()


def main():
    """主函数"""
    args = parse_args()
    
    print("="*60)
    print("DIFF_GRM DATASET ANALYZER")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Category: {args.category}")
    print(f"Pad token: {args.pad_token}")
    
    try:
        # 1. 加载配置
        config = load_config(args.config, args.category)
        
        # 2. 设置数据集和tokenizer
        print("\nSetting up dataset and tokenizer...")
        dataset_class = get_dataset(config['dataset'])
        dataset = dataset_class(config)
        print(f"Dataset: {dataset.__class__.__name__}, Items: {dataset.n_items}")
        
        tokenizer_class = get_tokenizer(config['model'])
        tokenizer = tokenizer_class(config, dataset)
        print(f"Tokenizer: {tokenizer.__class__.__name__}, Vocab size: {tokenizer.vocab_size}")
        
        # 3. Tokenize数据集
        print("\nTokenizing dataset...")
        datasets = dataset.split()
        tokenized_datasets = tokenizer.tokenize(datasets)
        
        # 4. 分析各个split
        results = {'config': config}
        
        for split_name in ['train', 'val', 'test']:
            if split_name in tokenized_datasets:
                stats = analyze_split(
                    tokenized_datasets[split_name],
                    split_name,
                    args.pad_token,
                    args.max_samples
                )
                results[split_name] = stats
                print_statistics(stats)
            else:
                print(f"\nWarning: Split '{split_name}' not found in dataset")
        
        # 5. 保存结果
        save_results(results, args)
        
        # 6. 绘制分布图（可选）
        if args.plot_distribution:
            plot_distribution(results)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 