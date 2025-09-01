#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
from typing import Optional
import argparse
from collections import defaultdict

import numpy as np

# ---- import your project code ----
# Resolve repo root as parent of this scripts/ directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from genrec.utils import get_config, get_dataset
from genrec.models.DIFF_GRM.tokenizer import DIFF_GRMTokenizer

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def infer_sent_emb_dim(model_id: str) -> int:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers 未安装，不能自动推断维度；请通过 --sent-emb-dim 指定。")
    m = SentenceTransformer(model_id, trust_remote_code=True)
    return m.get_sentence_embedding_dimension()


def tokens_to_codebooks(tokens, sid_offset, codebook_size):
    cbs = []
    for d, tok in enumerate(tokens):
        cbs.append(int(tok) - (sid_offset + d * codebook_size))
    return tuple(cbs)


def build_cb2items_map(tokenizer: DIFF_GRMTokenizer):
    cb2items = defaultdict(list)
    sid_offset = tokenizer.sid_offset
    K = tokenizer.codebook_size
    for item, toks in tokenizer.item2tokens.items():
        cb = tokens_to_codebooks(toks, sid_offset, K)
        cb2items[cb].append(item)
    return cb2items


def iterate_test_targets(dataset) -> list:
    splits = dataset.split()
    assert 'test' in splits, "dataset.split() 未返回 'test' split"
    test_ds = splits['test']
    if hasattr(test_ds, 'column_names') and 'item_seq' in test_ds.column_names:
        targets = [seq[-1] for seq in test_ds['item_seq']]
    else:
        targets = []
        for rec in test_ds:
            targets.append(rec['item_seq'][-1])
    return targets


def check_conflicts_once(
    dataset_name: str,
    n_digit: int,
    codebook_size: int,
    sid_quantizer: str,
    sent_emb_model: str,
    sent_emb_dim: int | None,
    sent_emb_pca: int,
    force_regen: bool,
    opq_use_gpu: bool,
    faiss_omp_num_threads: int,
    disable_opq: bool,
    rq_kmeans_niters: int,
    rq_kmeans_seed: int,
    category: Optional[str] = None,
    dump_writer: Optional[csv.DictWriter] = None,
) -> dict:

    cfg_overrides = {
        "dataset": dataset_name,
        "model": "DIFF_GRM",
        "n_digit": int(n_digit),
        "codebook_size": int(codebook_size),
        "sid_quantizer": sid_quantizer,
        "disable_opq": bool(disable_opq),
        "sent_emb_model": sent_emb_model,
        "sent_emb_dim": int(sent_emb_dim) if sent_emb_dim is not None else None,
        "sent_emb_pca": int(sent_emb_pca),
        "sent_emb_batch_size": 128,
        "normalize_after_pca": True,
        "metadata": "sentence",
        "num_proc": 1,
        "force_regenerate_opq": bool(force_regen),
        "opq_use_gpu": bool(opq_use_gpu),
        "opq_gpu_id": 0,
        "faiss_omp_num_threads": int(faiss_omp_num_threads),
        "rq_kmeans_niters": int(rq_kmeans_niters),
        "rq_kmeans_seed": int(rq_kmeans_seed),
    }

    if category:
        cfg_overrides["category"] = category

    # 注入简易 accelerator，使得不经训练管线也可安全调用 tokenizer/dataset/logger
    try:
        from types import SimpleNamespace
        cfg_overrides["accelerator"] = SimpleNamespace(is_main_process=True)
    except Exception:
        class _DummyAccel:
            is_main_process = True
        cfg_overrides["accelerator"] = _DummyAccel()

    if sid_quantizer == "rq_kmeans":
        cfg_overrides["sent_emb_pca"] = 0

    if cfg_overrides["sent_emb_dim"] is None:
        cfg_overrides["sent_emb_dim"] = infer_sent_emb_dim(sent_emb_model)

    # 统一检查：OPQ/PQ 的有效维度需可被 n_digit 整除
    if sid_quantizer == "opq_pq":
        d_eff = cfg_overrides["sent_emb_pca"] if cfg_overrides["sent_emb_pca"] > 0 else cfg_overrides["sent_emb_dim"]
        if d_eff % n_digit != 0:
            return {"skip_reason": f"effective_dim({d_eff}) % n_digit({n_digit}) != 0 for OPQ/PQ"}

    config = get_config("DIFF_GRM", dataset_name, config_file=None, config_dict=cfg_overrides)
    dataset_cls = get_dataset(dataset_name)
    dataset = dataset_cls(config)

    tokenizer = DIFF_GRMTokenizer(config, dataset)
    cb2items = build_cb2items_map(tokenizer)
    targets = iterate_test_targets(dataset)
    n_examples = len(targets)
    unique_targets = sorted(set(targets))
    n_unique_targets = len(unique_targets)

    item2cb = {}
    for item, toks in tokenizer.item2tokens.items():
        item2cb[item] = tokens_to_codebooks(toks, tokenizer.sid_offset, tokenizer.codebook_size)

    conflict_counts_per_example = []
    conflict_counts_per_unique_item = {}

    for tgt in targets:
        if tgt not in item2cb:
            conflict_counts_per_example.append(0)
            continue
        cb = item2cb[tgt]
        cnt = len(cb2items.get(cb, []))
        conflict_counts_per_example.append(max(0, cnt - 1))

    for tgt in unique_targets:
        if tgt not in item2cb:
            conflict_counts_per_unique_item[tgt] = 0
            continue
        cb = item2cb[tgt]
        cnt = len(cb2items.get(cb, []))
        conflict_counts_per_unique_item[tgt] = max(0, cnt - 1)
        # 可选：将发生冲突的 target 写出到 collisions 表
        if dump_writer is not None and cnt > 1:
            others = [x for x in cb2items.get(cb, []) if x != tgt]
            dump_writer.writerow({
                "dataset": dataset_name,
                "category": category or "",
                "n_digit": n_digit,
                "codebook_size": codebook_size,
                "sid_quantizer": sid_quantizer,
                "sent_emb_model": sent_emb_model,
                "sent_emb_pca": cfg_overrides["sent_emb_pca"],
                "target_item": tgt,
                "collision_size": cnt - 1,
                "sid": str(cb),
                "others": ";".join(others),
            })

    ex_conflicts = np.array(conflict_counts_per_example)
    u_conflicts = np.array(list(conflict_counts_per_unique_item.values()))

    result = {
        "n_digit": n_digit,
        "codebook_size": codebook_size,
        "sid_quantizer": sid_quantizer,
        "sent_emb_model": sent_emb_model,
        "sent_emb_dim": cfg_overrides["sent_emb_dim"],
        "sent_emb_pca": cfg_overrides["sent_emb_pca"],
        "n_test_examples": n_examples,
        "n_unique_targets": n_unique_targets,
        "examples_conflict_rate": float((ex_conflicts > 0).mean()) if n_examples > 0 else 0.0,
        "examples_mean_conflicts_if_conflicted": float(ex_conflicts[ex_conflicts > 0].mean()) if (ex_conflicts > 0).any() else 0.0,
        "examples_max_conflicts": int(ex_conflicts.max()) if ex_conflicts.size else 0,
        "unique_items_conflict_rate": float((u_conflicts > 0).mean()) if n_unique_targets > 0 else 0.0,
        "unique_items_mean_conflicts_if_conflicted": float(u_conflicts[u_conflicts > 0].mean()) if (u_conflicts > 0).any() else 0.0,
        "unique_items_max_conflicts": int(u_conflicts.max()) if u_conflicts.size else 0,
        "worst_examples": ";".join(
            f"{t}:{conflict_counts_per_unique_item[t]}"
            for t in sorted(conflict_counts_per_unique_item, key=lambda x: conflict_counts_per_unique_item[x], reverse=True)[:10]
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Check SID collision on LOO test targets (per config).")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., AmazonReviews2014).")
    parser.add_argument("--digits", type=int, nargs="+", default=[3, 4, 5], help="n_digit candidates.")
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--sid-quantizer", type=str, default="opq_pq", choices=["opq_pq", "rq_kmeans", "none"])
    parser.add_argument("--models", nargs="+", required=True, help="SentenceTransformer model ids.")
    parser.add_argument("--pca", type=int, nargs="+", default=[64], help="PCA dims (only for opq_pq).")
    parser.add_argument("--sent-emb-dim", type=int, default=None, help="OPTIONAL: if not provided, will be inferred.")
    parser.add_argument("--force-regen", action="store_true", help="Force regenerate quantization files.")
    parser.add_argument("--opq-use-gpu", action="store_true")
    parser.add_argument("--faiss-omp-num-threads", type=int, default=8)
    parser.add_argument("--disable-opq", action="store_true")
    parser.add_argument("--rq-kmeans-iters", type=int, default=20)
    parser.add_argument("--rq-kmeans-seed", type=int, default=1234)
    parser.add_argument("--out", type=str, default="sid_conflict_report.csv")
    parser.add_argument("--category", type=str, default=None, help="Optional dataset category, e.g. Toys_and_Games")
    parser.add_argument("--dump-collisions", type=str, default=None, help="Optional path to dump collided targets")

    args = parser.parse_args()

    rows = []
    header = [
        "dataset", "n_digit", "codebook_size", "sid_quantizer",
        "sent_emb_model", "sent_emb_dim", "sent_emb_pca",
        "n_test_examples", "n_unique_targets",
        "examples_conflict_rate", "examples_mean_conflicts_if_conflicted", "examples_max_conflicts",
        "unique_items_conflict_rate", "unique_items_mean_conflicts_if_conflicted", "unique_items_max_conflicts",
        "worst_examples", "skip_reason"
    ]

    combos = []
    if args.sid_quantizer == "opq_pq":
        for m in args.models:
            for pca in args.pca:
                for d in args.digits:
                    combos.append((m, pca, d))
    else:
        for m in args.models:
            for d in args.digits:
                combos.append((m, 0, d))

    # Optional collisions dump
    dump_writer = None
    if args.dump_collisions:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_collisions)), exist_ok=True)
        f_dump = open(args.dump_collisions, "w", newline="")
        dump_writer = csv.DictWriter(f_dump, fieldnames=[
            "dataset","category","n_digit","codebook_size","sid_quantizer",
            "sent_emb_model","sent_emb_pca","target_item","collision_size","sid","others"
        ])
        dump_writer.writeheader()

    for sent_model, pca_dim, nd in combos:
        result = None
        skip_reason = ""
        out_row = {
            "dataset": args.dataset,
            "n_digit": nd,
            "codebook_size": args.codebook_size,
            "sid_quantizer": args.sid_quantizer,
            "sent_emb_model": sent_model,
            "sent_emb_dim": "",
            "sent_emb_pca": pca_dim,
            "n_test_examples": "",
            "n_unique_targets": "",
            "examples_conflict_rate": "",
            "examples_mean_conflicts_if_conflicted": "",
            "examples_max_conflicts": "",
            "unique_items_conflict_rate": "",
            "unique_items_mean_conflicts_if_conflicted": "",
            "unique_items_max_conflicts": "",
            "worst_examples": "",
            "skip_reason": "",
        }
        try:
            result = check_conflicts_once(
                dataset_name=args.dataset,
                n_digit=nd,
                codebook_size=args.codebook_size,
                sid_quantizer=args.sid_quantizer,
                sent_emb_model=sent_model,
                sent_emb_dim=args.sent_emb_dim,
                sent_emb_pca=pca_dim,
                force_regen=args.force_regen,
                opq_use_gpu=args.opq_use_gpu,
                faiss_omp_num_threads=args.faiss_omp_num_threads,
                disable_opq=args.disable_opq,
                rq_kmeans_niters=args.rq_kmeans_iters,
                rq_kmeans_seed=args.rq_kmeans_seed,
                category=args.category,
                dump_writer=dump_writer,
            )
        except Exception as e:
            skip_reason = str(e)

        if result is not None and "skip_reason" in result:
            skip_reason = result["skip_reason"]

        if result is not None and not skip_reason:
            out_row.update({
                "sent_emb_dim": result.get("sent_emb_dim"),
                "sent_emb_pca": result.get("sent_emb_pca"),
                "n_test_examples": result.get("n_test_examples"),
                "n_unique_targets": result.get("n_unique_targets"),
                "examples_conflict_rate": result.get("examples_conflict_rate"),
                "examples_mean_conflicts_if_conflicted": result.get("examples_mean_conflicts_if_conflicted"),
                "examples_max_conflicts": result.get("examples_max_conflicts"),
                "unique_items_conflict_rate": result.get("unique_items_conflict_rate"),
                "unique_items_mean_conflicts_if_conflicted": result.get("unique_items_mean_conflicts_if_conflicted"),
                "unique_items_max_conflicts": result.get("unique_items_max_conflicts"),
                "worst_examples": result.get("worst_examples"),
            })
            print(
                f"[n_digit={nd} | model={sent_model} | pca={pca_dim}] "
                f"Ex-ConflictRate={out_row['examples_conflict_rate']:.4f}, "
                f"Uniq-ConflictRate={out_row['unique_items_conflict_rate']:.4f}, "
                f"MaxConflicts(uniq)={out_row['unique_items_max_conflicts']}"
            )
        else:
            out_row["skip_reason"] = skip_reason or "unknown"
            print(f"[n_digit={nd} | model={sent_model} | pca={pca_dim}] SKIP: {out_row['skip_reason']}")

        rows.append(out_row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved report to: {os.path.abspath(args.out)}")

    if args.dump_collisions and dump_writer is not None:
        f_dump.close()
        print(f"Saved collisions to: {os.path.abspath(args.dump_collisions)}")


if __name__ == "__main__":
    main()

