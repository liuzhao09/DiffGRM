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
CUDA_VISIBLE_DEVICES=2 python main.py \
  --category=Sports_and_Outdoors \
  --train_batch_size=1024 \
  --model=DIFF_GRM \
  --n_digit=4 \
  --masking_strategy=guided \
  --guided_refresh_each_step=false \
  --guided_select=least \
  --guided_conf_metric=msp \
  --encoder_n_layer=1 \
  --decoder_n_layer=4 \
  --n_head=4 \
  --n_embd=256 \
  --n_inner=1024 \
  --train_sliding=true \
  --min_hist_len=2 \
  --eval_start_epoch=20 \
  --lr=0.003 \
  --label_smoothing=0.1 \
  --sent_emb_model="sentence-transformers/sentence-t5-base" \
  --sent_emb_dim=768 \
  --sent_emb_pca=256 \
  --sent_emb_batch_size=256 \
  --normalize_after_pca=true \
  --force_regenerate_opq=true \
  --share_decoder_output_embedding=true > runs/sports/t5_pca256_guided_least_0_msp_1e4d_256dim_xxx_xxx_xxx.txt 2>&1 &
```


### Beauty

```
CUDA_VISIBLE_DEVICES=5 python main.py \
  --category=Beauty \
  --train_batch_size=1024 \
  --model=DIFF_GRM \
  --n_digit=4 \
  --masking_strategy=guided \
  --guided_refresh_each_step=false \
  --guided_select=least \
  --guided_conf_metric=msp \
  --encoder_n_layer=1 \
  --decoder_n_layer=4 \
  --n_head=4 \
  --n_embd=256 \
  --n_inner=1024 \
  --train_sliding=true \
  --min_hist_len=2 \
  --eval_start_epoch=20 \
  --lr=0.01 \
  --label_smoothing=0.2 \
  --sent_emb_model=sentence-transformers/sentence-t5-base \
  --sent_emb_dim=768 \
  --sent_emb_pca=256 \
  --sent_emb_batch_size=256 \
  --normalize_after_pca=true \
  --force_regenerate_opq=true \
  --share_decoder_output_embedding=true > runs/beauty/ls02_t5_pca256_guided_least_0_msp_1e4d_256dim_xxx_xxx_xxx.txt 2>&1 &
```

### Toys and Games

```
CUDA_VISIBLE_DEVICES=0 python main.py \
  --category=Toys_and_Games \
  --train_batch_size=1024 \
  --model=DIFF_GRM \
  --n_digit=4 \
  --masking_strategy=guided \
  --guided_refresh_each_step=false \
  --guided_select=least \
  --guided_conf_metric=msp \
  --encoder_n_layer=1 \
  --decoder_n_layer=4 \
  --n_head=8 \
  --n_embd=1024 \
  --n_inner=1024 \
  --train_sliding=true \
  --min_hist_len=2 \
  --eval_start_epoch=10 \
  --lr=0.003 \
  --label_smoothing=0.15 \
  --sent_emb_model="sentence-transformers/sentence-t5-base" \
  --sent_emb_dim=768 \
  --sent_emb_pca=256 \
  --sent_emb_batch_size=256 \
  --normalize_after_pca=true \
  --force_regenerate_opq=true \
  --share_decoder_output_embedding=true > runs/toys/h8_ls015_t5_pca256_guided_least_0_msp_1e4d_1024dim_9_4_0.txt 2>&1 &
```


