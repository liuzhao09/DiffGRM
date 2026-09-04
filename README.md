# DiffGRM: Diffusion-based Generative Recommendation Model

[![The Web Conference 2026](https://img.shields.io/badge/The_Web_Conference-2026-blue)](https://doi.org/10.1145/3774904.3792156)
[![arXiv](https://img.shields.io/badge/arXiv-2510.21805-b31b1b.svg)](https://arxiv.org/abs/2510.21805)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Official PyTorch implementation of **“DiffGRM: Diffusion-based Generative Recommendation Model,”** accepted by **The ACM Web Conference 2026 (WWW 2026)**.

DiffGRM replaces left-to-right autoregressive Semantic-ID generation with masked discrete diffusion. It enables bidirectional interaction among SID digits and supports confidence-guided, any-order parallel denoising for recommendation.

## Paper

- ACM Digital Library: [https://doi.org/10.1145/3774904.3792156](https://doi.org/10.1145/3774904.3792156)
- arXiv: [https://arxiv.org/abs/2510.21805](https://arxiv.org/abs/2510.21805)
- Venue: The ACM Web Conference 2026
- Pages: 5853–5864

## Environment Setup

Python 3.10 and a CUDA-capable GPU are recommended.

```bash
git clone https://github.com/liuzhao09/DiffGRM.git
cd DiffGRM

conda create -n diffgrm python=3.10 -y
conda activate diffgrm
pip install -r requirements.txt
```

## Data

The current implementation evaluates on the 5-core subsets of the Amazon Reviews 2014 dataset. Supported categories include `Sports_and_Outdoors`, `Beauty`, and `Toys_and_Games`.

Raw review and metadata files are downloaded on first use from the Stanford SNAP Amazon dataset host and cached under `cache/`. Processed data, item mappings, sentence embeddings, OPQ state, and Semantic-ID mappings are generated locally as needed.

The dataset is not distributed by this repository and is not covered by the DiffGRM license. Users are responsible for reviewing and complying with the source dataset's applicable terms.

## Reproduction

Configuration values are loaded from:

1. `genrec/default.yaml`
2. `genrec/datasets/AmazonReviews2014/config.yaml`
3. `genrec/models/DIFF_GRM/config.yaml`
4. command-line overrides

The following commands reproduce the main DiffGRM settings. Create the output directories before redirecting logs:

```bash
mkdir -p runs/sports runs/beauty runs/toys
```

### Sports and Outdoors

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
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
  --sent_emb_model=sentence-transformers/sentence-t5-base \
  --sent_emb_dim=768 \
  --sent_emb_pca=256 \
  --sent_emb_batch_size=256 \
  --normalize_after_pca=true \
  --force_regenerate_opq=true \
  --share_decoder_output_embedding=true \
  > runs/sports/diffgrm.log 2>&1
```

### Beauty

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
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
  --share_decoder_output_embedding=true \
  > runs/beauty/diffgrm.log 2>&1
```

### Toys and Games

```bash
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
  --sent_emb_model=sentence-transformers/sentence-t5-base \
  --sent_emb_dim=768 \
  --sent_emb_pca=256 \
  --sent_emb_batch_size=256 \
  --normalize_after_pca=true \
  --force_regenerate_opq=true \
  --share_decoder_output_embedding=true \
  > runs/toys/diffgrm.log 2>&1
```

Set `--force_regenerate_opq=false` after the relevant cache has been generated if you want subsequent runs to reuse it.

## Checkpoints and Other Artifacts

No checkpoints, prepared datasets, tokenizer caches, OPQ caches, or Semantic-ID mappings are currently distributed in this repository.

Unless a specific release says otherwise, project-created checkpoints, tokenizer or OPQ caches, and Semantic-ID mappings released by Kuaishou Technology as part of DiffGRM are provided under the same [CC BY-NC 4.0](LICENSE.md) terms as the code. This grant applies only to rights held by Kuaishou Technology and does not replace or override the licenses or terms governing any underlying datasets, pretrained models, or other third-party material.

Artifacts obtained or generated by users are subject to all applicable upstream terms. An artifact that has not been publicly released should be treated as unavailable and not licensed for redistribution by this repository.

## Citation

If you use this code or the released DiffGRM artifacts in academic work, please cite:

```bibtex
@inproceedings{liu2026diffgrm,
  author    = {Zhao Liu and Yichen Zhu and Yiqing Yang and Xiao Lv and
               Guoping Tang and Rui Huang and Qiang Luo and Ruiming Tang and
               Guorui Zhou},
  title     = {DiffGRM: Diffusion-based Generative Recommendation Model},
  booktitle = {Proceedings of the ACM Web Conference 2026},
  pages     = {5853--5864},
  year      = {2026},
  doi       = {10.1145/3774904.3792156}
}
```

Machine-readable citation metadata is also available in [`CITATION.cff`](CITATION.cff).

## Acknowledgements

DiffGRM is developed from the codebase of [RPG: Generating Long Semantic IDs in Parallel for Recommendation](https://github.com/facebookresearch/RPG_KDD2025). We thank its authors for releasing their implementation. The original Meta copyright notices have been retained in upstream-derived source files; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Unless otherwise noted, this repository is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE.md) (**CC BY-NC 4.0**).

You may share and adapt the licensed material for non-commercial purposes, provided that you give appropriate attribution, link to the license, and indicate whether changes were made. Commercial use is not granted by this public license. Patent and trademark rights are not licensed.

Copyright © 2025–2026 Kuaishou Technology. Portions copyright © Meta Platforms, Inc. and affiliates.

Third-party libraries, datasets, pretrained models, and other external materials remain subject to their own licenses and terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and scope details.

## Contributing and Contact

Contributions are welcome under the terms described in [`CONTRIBUTING.md`](CONTRIBUTING.md). For research questions, please open a GitHub issue or contact the corresponding authors listed in the paper.

