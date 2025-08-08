# DiffGM

This repository provides the code for implementing DiffGM described in our paper.

## Introduction

生成式推荐是新兴的序列推荐范式，主流的生成式推荐方式大多采用残差量化（RQ）的方式将用户历史行为（items）编码为离散的多级 sid，并用基于自回归的 Transformer 架构去生成多级 sid。但是基于 RQ 的 tokenizer 存在明显的语义沙漏问题【C1】，前面层包含主要信息，约到后面信息越少，这种现象导致扩展层数不但不能增加推荐精度，反而会增加推理耗时。

而采用平行编码方式能够保证没有语义沙漏问题，每层 sid 包含的信息均衡，并且在流式数据下也验证了 4 层 扩展到 8 层在精度有提升。然而自回归架构必须顺序依赖【C2】，限制了平行语义编码的 sid 的能力，这导致在 codebook 层数相同的情况下，采用残差编码的方式一直比采用平行编码的方式效果好。

为此我们借鉴 Gemini Diffusion 在 LLM 领域的启发，其打破了自回归范式的限制，能够双向捕获 token 之间的关系，并能够高效的并行推理，尽管这种方式有个很大的弊端就是上下文长度固定，这导致这种方式与自回归模型理论上可以无限长的上下文还有很大差距，在现有的 LLM 应用中也尚未普及。

庆幸的是，在生成式推荐范式下，固定数量的 sid 去表示一个 item 是比较合适，故此 Diffusion Model 在 LLM 领域的上下文长度固定的问题并不会困扰我们，反而为我们平行语义编码的 sid 提供了感知双向关系、灵活并行生成赋能。


## Installation

### Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd DiffGM
```

2. **Create a conda environment (recommended):**
```bash
conda create -n diffgm python=3.10 -y
conda activate diffgm
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Requirements

The main dependencies include:
- **PyTorch** (2.6.0) - Deep learning framework
- **Transformers** (4.53.2) - Hugging Face transformers library
- **Sentence-Transformers** (3.3.1) - For sentence embeddings
- **FAISS** (1.11.0) - For efficient similarity search
- **Accelerate** (0.31.0) - For distributed training
- **Wandb** (0.19.0) - For experiment tracking

For GPU support, make sure you have CUDA installed. The requirements.txt includes CUDA 12.4 compatible versions.

### System Requirements

- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU training)
- **Memory**: At least 16GB RAM (32GB recommended for large datasets)
- **Storage**: At least 50GB free space for datasets and models

## Quick Start

Run the following command to start training the model with a specified category:

```
CUDA_VISIBLE_DEVICES=0 python main.py --category=Sports_and_Outdoors
```

Available categories:
* `Sports_and_Outdoors`
* `Beauty`
* `Toys_and_Games`
* `CDs_and_Vinyl`

Note that:
1. The datasets will be automatically downloaded once the `category` argument is specified.
2. All hyperparameters can be specified via command line arguments. Please refer to:
    * `genrec/default.yaml`
    * `genrec/datasets/AmazonReviews2014/config.yaml`
    * `genrec/models/RPG/config.yaml`

## Reproduction

### Sports and Outdoors


```
CUDA_VISIBLE_DEVICES=0 nohup python main.py \
    --category=Sports_and_Outdoors \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=10 \
    --n_edges=100 \
    --propagation_steps=2 > runs/sports/7_20_13.txt 2>&1 &
```

CUDA_VISIBLE_DEVICES=0 python genrec/models/DIFF_GRM/standalone_sid_builder.py \
  --dataset AmazonReviews2014 \
  --category Sports_and_Outdoors \
  --n_digit 8 \
  --codebook_size 256 \
  --sent_emb_model sentence-transformers/sentence-t5-base \
  --sent_emb_dim 768 \
  --sent_emb_pca 128 \
  --sent_emb_batch_size 128 \
  --faiss_omp_num_threads 32 \
  --embed_use_gpu \
  --force


CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors \
    --train_batch_size=1024 \
    --model=DIFF_GRM \
    --n_digit=4 \
    --masking_strategy=sequential \
    --sequential_paths=1 \
    --encoder_n_layer=2 \
    --decoder_n_layer=4 \
    --n_head=4 \
    --n_embd=256 \
    --n_inner=1024 \
    --train_sliding=true \
    --min_hist_len=2 \
    --eval_start_epoch=20 \
    --lr=0.003 \
    --label_smoothing=0.1 \
    --sent_emb_pca=256 \
    --share_decoder_output_embedding=true \
    --temperature=0.03 > runs/sports/4layer_pca256_1sequential_2e4d_256dim_diff_8_7_12.txt 2>&1 &
    

CUDA_VISIBLE_DEVICES=0 python genrec/models/DIFF_GRM/infer.py \
    --checkpoint saved/AmazonReviews2014_Aug-04-2025_16-50/pytorch_model.bin \
    --category Sports_and_Outdoors \
    --top_k 10 \
    --beam_act 128 \
    --beam_max 128 \
    --dedup_strategy simple \
    --batch_size 32 \
    --n_digit=4 \
    --encoder_n_layer=4 \
    --decoder_n_layer=4 \
    --n_head=4 \
    --n_embd=512 \
    --n_inner=1024 \
    --print_duplicates \
    --share_embeddings=true \
    --debug > runs/sports/infer_simple_8times_beam128_8_5_7.txt 2>&1 &



### Beauty

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Beauty \
    --lr=0.01 \
    --temperature=0.03 \
    --n_codebook=32 \
    --num_beams=10 \
    --n_edges=100 \
    --propagation_steps=3
```



CUDA_VISIBLE_DEVICES=1 python main.py \
  --category=Sports_and_Outdoors \
  --train_batch_size=1024 \
  --model=DIFF_GRM \
  --n_digit=4 \
  --masking_strategy=sequential \
  --sequential_paths=1 \
  --encoder_n_layer=2 \
  --decoder_n_layer=4 \
  --n_head=4 \
  --n_embd=256 \
  --n_inner=1024 \
  --train_sliding=true \
  --min_hist_len=2 \
  --eval_start_epoch=25 \
  --lr=0.003 \
  --label_smoothing=0.1 \
  --sent_emb_model="Alibaba-NLP/gte-large-en-v1.5" \
  --sent_emb_dim=1024 \
  --sent_emb_pca=256 \
  --sent_emb_batch_size=256 \
  --normalize_after_pca=true \
  --force_regenerate_opq=true \
  --share_decoder_output_embedding=true > runs/sports/4layer_gtepca256_seq1_2e4d_256dim_8_8_16.txt 2>&1 &



CUDA_VISIBLE_DEVICES=3 python main.py \
    --category=Beauty \
    --train_batch_size=1024 \
    --model=DIFF_GRM \
    --n_digit=4 \
    --masking_strategy=sequential \
    --sequential_paths=1 \
    --encoder_n_layer=2 \
    --decoder_n_layer=4 \
    --n_head=4 \
    --n_embd=256 \
    --n_inner=1024 \
    --train_sliding=true \
    --min_hist_len=2 \
    --eval_start_epoch=20 \
    --lr=0.01 \
    --label_smoothing=0.1 \
    --sent_emb_pca=128 \
    --share_decoder_output_embedding=true \
    --temperature=0.03 > runs/beauty/4layer_pca128_1sequential_2e4d_256dim_diff_8_8_15.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 python genrec/models/DIFF_GRM/infer.py \
    --checkpoint saved/AmazonReviews2014_Aug-05-2025_07-53/pytorch_model.bin \
    --category Beauty \
    --top_k 10 \
    --beam_act 128 \
    --beam_max 128 \
    --dedup_strategy simple \
    --batch_size 32 \
    --n_digit=4 \
    --encoder_n_layer=4 \
    --decoder_n_layer=4 \
    --n_head=4 \
    --n_embd=512 \
    --n_inner=1024 \
    --print_duplicates \
    --debug > runs/beauty/infer_simple_4times_rand_beam128_8_5_10.txt 2>&1 &

### Toys and Games

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Toys_and_Games \
    --lr=0.003 \
    --temperature=0.03 \
    --n_codebook=16 \
    --num_beams=10 \
    --n_edges=50 \
    --propagation_steps=5
```


CUDA_VISIBLE_DEVICES=2 python main.py \
    --category=Toys_and_Games \
    --train_batch_size=1024 \
    --model=DIFF_GRM \
    --n_digit=4 \
    --mask_probs="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --encoder_n_layer=4 \
    --decoder_n_layer=4 \
    --n_head=4 \
    --n_embd=512 \
    --n_inner=1024 \
    --eval_start_epoch=10 \
    --lr=0.003 \
    --share_decoder_output_embedding=true \
    --temperature=0.03 > runs/toys/4layer_10times_4e4d_512dim_diff_8_5_14.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python main.py \
    --category=Toys_and_Games \
    --train_batch_size=1024 \
    --model=DIFF_GRM \
    --n_digit=4 \
    --masking_strategy=random \
    --mask_probs="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --encoder_n_layer=2 \
    --decoder_n_layer=4 \
    --n_head=4 \
    --n_embd=256 \
    --n_inner=1024 \
    --train_sliding=true \
    --min_hist_len=2 \
    --eval_start_epoch=20 \
    --lr=0.003 \
    --label_smoothing=0.1 \
    --sent_emb_pca=128 \
    --share_decoder_output_embedding=true \
    --temperature=0.03 > runs/toys/4layer_pca128_10rand_2e4d_256dim_diff_8_8_11.txt 2>&1 &
    

### CDs and Vinyl

```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=CDs_and_Vinyl \
    --lr=0.001 \
    --temperature=0.03 \
    --n_codebook=64 \
    --num_beams=10 \
    --n_edges=500 \
    --propagation_steps=3
```

CUDA_VISIBLE_DEVICES=2 python main.py \
    --category=CDs_and_Vinyl \
    --train_batch_size=256 \
    --model=DIFF_GRM \
    --n_digit=4 \
    --mask_probs="1.0,0.75,0.5,0.25" \
    --encoder_n_layer=2 \
    --decoder_n_layer=4 \
    --n_head=4 \
    --n_embd=512 \
    --n_inner=1024 \
    --lr=0.001 \
    --temperature=0.03 > runs/cds/4layer_2e4d_512dim_diff_8_3_15.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python main.py \
    --category=CDs_and_Vinyl \
    --train_batch_size=1024 \
    --model=DIFF_GRM \
    --n_digit=4 \
    --masking_strategy=sequential \
    --sequential_paths=1 \
    --encoder_n_layer=2 \
    --decoder_n_layer=4 \
    --n_head=4 \
    --n_embd=256 \
    --n_inner=1024 \
    --train_sliding=true \
    --min_hist_len=2 \
    --eval_start_epoch=20 \
    --lr=0.001 \
    --sent_emb_pca=128 \
    --share_decoder_output_embedding=true \
    --temperature=0.03 > runs/cds/4layer_pca128_1sequential_2e4d_256dim_diff_8_6_23.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python genrec/models/DIFF_GRM/infer.py \
    --checkpoint saved/AmazonReviews2014_Aug-03-2025_15-05/pytorch_model.bin \
    --category CDs_and_Vinyl \
    --top_k 10 \
    --beam_act 512 \
    --beam_max 512 \
    --batch_size 16 \
    --print_duplicates \
    --debug > runs/cds/infer_4layer_2e4d_512dim_diff_8_4_11.txt 2>&1 &


