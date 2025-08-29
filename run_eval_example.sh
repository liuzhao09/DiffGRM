#!/bin/bash
set -euo pipefail

# 切到脚本所在目录（即仓库根）
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
export PYTHONPATH="$DIR:${PYTHONPATH:-}"

echo "🚀 Working directory: $(pwd)"
echo "🚀 PYTHONPATH: $PYTHONPATH"

# 示例：使用eval_checkpoint.py进行推理
# 这个脚本展示了如何使用训练好的checkpoint进行测试

echo "🚀 DIFF_GRM Checkpoint Evaluation Examples"
echo "=========================================="

# 设置基本参数
MODEL="DIFF_GRM"
DATASET="AmazonReviews2014"
CATEGORY="Toys_and_Games"
CHECKPOINT="saved/AmazonReviews2014_Aug-29-2025_02-52-21/pytorch_model.bin"


echo ""
echo "📋 Both confidence and random modes"
echo "-----------------------------------------------"

python eval_checkpoint.py \
    --model=$MODEL \
    --dataset=$DATASET \
    --category=$CATEGORY \
    --checkpoint="$CHECKPOINT" \
    --prefer_ckpt_arch \
    --beam_search_modes='["confidence", "random"]' \
    --vectorized_beam_search='{
        "top_k_final": 10,
        "dedup_strategy": "weighted",
        "test": {"beam_act": 256, "beam_max": 256},
        "neg_inf_fp32": -1000000000.0,
        "neg_inf_fp16": -65504.0
    }' \
    --random_beam='{"beam_act": 64, "beam_max": 64, "seed": 42}' \
    --eval_batch_size=32 \
    --train_sliding=true \
    --min_hist_len=2 \
    --max_hist_len=50 \
    --n_digit=4 \
    --codebook_size=256 \
    --sent_emb_model="Alibaba-NLP/gte-large-en-v1.5" \
    --sent_emb_dim=1024 \
    --sent_emb_pca=256 \
    --normalize_after_pca=true \
    --sid_quantizer="opq_pq" \
    --disable_opq=false \
    --force_regenerate_opq=false \
    --output_file="results/eval_results_$(date +%Y%m%d_%H%M%S).json"

