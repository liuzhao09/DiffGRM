# DiffGM

This repository provides the code for implementing DiffGM described in our paper.

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


## Reproduction


### Sports and Outdoors

```
CUDA_VISIBLE_DEVICES=0 python main.py \
  --category=Sports_and_Outdoors \
  --train_batch_size=1024 \
  --model=DIFF_GRM \
  --n_digit=4 \
  --masking_strategy=sequential \
  --sequential_paths=8 \
  --guided_select=least \
  --encoder_n_layer=2 \
  --decoder_n_layer=4 \
  --n_head=4 \
  --n_embd=256 \
  --n_inner=1024 \
  --train_sliding=true \
  --min_hist_len=2 \
  --eval_start_epoch=15 \
  --lr=0.003 \
  --label_smoothing=0.1 \
  --sid_quantizer=rq_kmeans \
  --sent_emb_model="sentence-transformers/sentence-t5-base" \
  --sent_emb_dim=768 \
  --sent_emb_pca=0 \
  --sent_emb_batch_size=256 \
  --normalize_after_pca=true \
  --force_regenerate_opq=true \
  --share_decoder_output_embedding=true > runs/sports/4layer_rq_8seq_2e4d_256dim_8_14_13.txt 2>&1 &
```


### Beauty

```
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
    --sent_emb_model="BAAI/bge-large-en-v1.5" \
    --sent_emb_dim=1024 \
    --sent_emb_pca=256 \
    --sent_emb_batch_size=256 \
    --normalize_after_pca=true \
    --force_regenerate_opq=true \
    --share_decoder_output_embedding=true > runs/beauty/4layer_bgepca128_1sequential_2e4d_256dim_diff_8_8_19.txt 2>&1 &
```

### Toys and Games

```
CUDA_VISIBLE_DEVICES=1 python main.py \
  --category=Toys_and_Games \
  --train_batch_size=1024 \
  --model=DIFF_GRM \
  --n_digit=4 \
  --masking_strategy=guided \
  --guided_refresh_each_step=true \
  --guided_select=least \
  --encoder_n_layer=2 \
  --decoder_n_layer=4 \
  --n_head=4 \
  --n_embd=256 \
  --n_inner=1024 \
  --train_sliding=true \
  --min_hist_len=2 \
  --eval_start_epoch=25 \
  --lr=0.003 \
  --label_smoothing=0.02 \
  --sent_emb_model="sentence-transformers/sentence-t5-base" \
  --sent_emb_dim=768 \
  --sent_emb_pca=256 \
  --sent_emb_batch_size=256 \
  --normalize_after_pca=true \
  --force_regenerate_opq=true \
  --share_decoder_output_embedding=true > runs/toys/4layer_bgepca256_guided_least_2e4d_256dim_8_14_12.txt 2>&1 &
```


