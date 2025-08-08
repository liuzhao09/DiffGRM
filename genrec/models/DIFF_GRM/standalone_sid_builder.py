# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import math
import json
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer

# 添加项目根目录到路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from genrec.utils import (
    get_config,
    init_seed,
    init_logger,
    init_device,
    get_dataset,
    get_tokenizer
)
from accelerate import Accelerator


class StandaloneSIDBuilder:
    """
    独立的SID构建器，可以单独运行生成sid2item和item2sid文件
    包含分布分析和统计输出
    """
    
    def __init__(self, config: dict, dataset, force_regenerate=False):
        self.config = config
        self.dataset = dataset
        self.force_regenerate = force_regenerate
        
        # 基本配置
        self.n_digit = config['n_digit']
        self.codebook_size = config['codebook_size']
        self.n_codebook_bits = self._get_codebook_bits(self.codebook_size)
        
        # 构建index_factory - 支持OPQ开关
        use_opq = not config.get('disable_opq', False)
        if use_opq:
            self.index_factory = f'OPQ{self.n_digit},IVF1,PQ{self.n_digit}x{self.n_codebook_bits}'
        else:
            self.index_factory = f'IVF1,PQ{self.n_digit}x{self.n_codebook_bits}'
        
        # 特殊token
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.sid_offset = 3
        
        # 路径配置
        self.cache_dir = self._get_cache_dir()
        self.sent_emb_path = os.path.join(
            self.cache_dir,
            f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
        )
        self.sem_ids_path = os.path.join(
            self.cache_dir,
            f'{os.path.basename(self.config["sent_emb_model"])}_{self.index_factory}.sem_ids'
        )
        self.suffix = f'_{self.n_digit}d'
        self.item_id2tokens_path = os.path.join(self.cache_dir, f'item_id2tokens{self.suffix}.npy')
        self.tokens2item_path = os.path.join(self.cache_dir, f'tokens2item{self.suffix}.pkl')
        
        # 统计信息
        self.stats = {
            'total_items': 0,
            'legal_items': 0,
            'unique_sids': 0,
            'sid_distribution': defaultdict(int),
            'digit_distribution': defaultdict(lambda: defaultdict(int)),
            'collision_stats': defaultdict(int)
        }
    
    def _get_codebook_bits(self, n_codebook):
        """计算codebook位数"""
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)
    
    def _get_cache_dir(self):
        """获取缓存目录"""
        dataset_name = self.dataset.__class__.__name__
        if hasattr(self.dataset, 'category') and self.dataset.category:
            cache_dir = os.path.join(self.dataset.cache_dir, 'processed')
        else:
            cache_dir = os.path.join('data', dataset_name, 'processed')
        return cache_dir
    
    def _encode_sent_emb(self):
        """编码句子嵌入"""
        print(f"[SID_BUILDER] Encoding sentence embeddings...")
        print(f"[SID_BUILDER] Model: {self.config['sent_emb_model']}")
        print(f"[SID_BUILDER] Batch size: {self.config['sent_emb_batch_size']}")
        
        meta_sentences = []
        for i in range(1, self.dataset.n_items):
            meta_sentences.append(self.dataset.item2meta[self.dataset.id_mapping['id2item'][i]])
        
        print(f"[SID_BUILDER] Total sentences: {len(meta_sentences)}")
        
        if 'sentence-transformers' in self.config['sent_emb_model']:
            # 根据是否使用GPU设置设备
            device = 'cuda' if self.config.get('embed_use_gpu', False) and torch.cuda.is_available() else 'cpu'
            sent_emb_model = SentenceTransformer(self.config['sent_emb_model'], device=device)
            
            sent_embs = sent_emb_model.encode(
                meta_sentences,
                convert_to_numpy=True,
                batch_size=self.config['sent_emb_batch_size'],
                show_progress_bar=True,
                device=device
            )
        else:
            raise ValueError(f"Unsupported embedding model: {self.config['sent_emb_model']}")
        
        print(f"[SID_BUILDER] Sentence embeddings shape: {sent_embs.shape}")
        
        # 保存句子嵌入
        os.makedirs(os.path.dirname(self.sent_emb_path), exist_ok=True)
        sent_embs.tofile(self.sent_emb_path)
        print(f"[SID_BUILDER] Saved sentence embeddings to {self.sent_emb_path}")
        
        return sent_embs
    
    def _get_items_for_training(self):
        """获取训练用的商品"""
        print(f"[SID_BUILDER] Getting items for training...")
        
        items_for_training = set()
        split_data = self.dataset.split()
        
        if 'train' in split_data:
            train_dataset = split_data['train']
            if hasattr(train_dataset, 'column_names') and 'item_seq' in train_dataset.column_names:
                for item_seq in train_dataset['item_seq']:
                    if isinstance(item_seq, (list, tuple)):
                        items_for_training.update(item_seq)
                    else:
                        items_for_training.add(item_seq)
        
        n_sent_embs = self.dataset.n_items - 1
        print(f"[SID_BUILDER] Items for training: {len(items_for_training)} of {n_sent_embs}")
        
        mask = np.zeros(n_sent_embs, dtype=bool)
        for item in items_for_training:
            item_id = self.dataset.item2id[item]
            if 1 <= item_id < self.dataset.n_items:
                mask[item_id - 1] = True
        
        print(f"[SID_BUILDER] Training mask shape: {mask.shape}, True count: {np.sum(mask)}")
        return mask
    
    def _generate_semantic_id_opq(self, sent_embs, train_mask):
        """使用OPQ生成语义ID"""
        import faiss
        
        print(f"[SID_BUILDER] Generating semantic IDs using OPQ...")
        print(f"[SID_BUILDER] Index factory: {self.index_factory}")
        print(f"[SID_BUILDER] Sent embeddings shape: {sent_embs.shape}")
        print(f"[SID_BUILDER] Training mask True count: {np.sum(train_mask)}")
        
        if self.config['opq_use_gpu']:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56
            print(f"[SID_BUILDER] Using GPU: {self.config['opq_gpu_id']}")
        
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        print(f"[SID_BUILDER] FAISS threads: {self.config['faiss_omp_num_threads']}")
        
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        
        print(f"[SID_BUILDER] Training index...")
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)
        
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)
        
        # ★！！！关键补丁：构建direct map以支持reconstruct ★
        # 需要在真正的IVF索引上调用make_direct_map，而不是IndexPreTransform
        ivf_index = faiss.extract_index_ivf(index)
        ivf_index.make_direct_map()
        
        print(f"[SID_BUILDER] Index training completed")
        
        # 提取PQ codes - 兼容OPQ和纯PQ两种模式
        base_index = faiss.downcast_index(
            index.index if hasattr(index, "index") else index
        )
        invlists = faiss.extract_index_ivf(base_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)
        
        print(f"[SID_BUILDER] PQ codes shape: {pq_codes.shape}")
        
        # 转换为语义ID
        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for i in range(self.n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)
        
        pq_codes = np.array(faiss_sem_ids)
        print(f"[SID_BUILDER] Semantic IDs shape: {pq_codes.shape}")
        
        # 创建item2sem_ids映射
        item2sem_ids = {}
        for i in range(pq_codes.shape[0]):
            item = self.dataset.id_mapping['id2item'][i + 1]
            item2sem_ids[item] = tuple(pq_codes[i].tolist())
        
        # 保存语义ID
        os.makedirs(os.path.dirname(self.sem_ids_path), exist_ok=True)
        with open(self.sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)
        
        print(f"[SID_BUILDER] Saved semantic IDs to {self.sem_ids_path}")
        
        # ------- quality check -------
        split_data = self.dataset.split()
        val_items = set()
        test_items = set()
        
        if 'val' in split_data:
            val_dataset = split_data['val']
            if hasattr(val_dataset, 'column_names') and 'item_seq' in val_dataset.column_names:
                for item_seq in val_dataset['item_seq']:
                    if isinstance(item_seq, (list, tuple)):
                        val_items.update(item_seq)
                    else:
                        val_items.add(item_seq)
        
        if 'test' in split_data:
            test_dataset = split_data['test']
            if hasattr(test_dataset, 'column_names') and 'item_seq' in test_dataset.column_names:
                for item_seq in test_dataset['item_seq']:
                    if isinstance(item_seq, (list, tuple)):
                        test_items.update(item_seq)
                    else:
                        test_items.add(item_seq)
        
        def to_mask(item_set):
            m = np.zeros(self.dataset.n_items-1, dtype=bool)
            for it in item_set:
                iid = self.dataset.item2id.get(it, 0)
                if 1 <= iid < self.dataset.n_items: 
                    m[iid-1] = True
            return m
        
        self._check_reconstruction_quality(index, sent_embs, to_mask(val_items), tag="val")
        self._check_reconstruction_quality(index, sent_embs, to_mask(test_items), tag="test")
        
        return item2sem_ids
    
    def _sem_ids_to_tokens(self, item2sem_ids):
        """将语义ID转换为token"""
        print(f"[SID_BUILDER] Converting semantic IDs to tokens...")
        
        item2tokens = {}
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            # 每个digit的codebook ID加上对应的offset
            tokens = [t + self.sid_offset + d * self.codebook_size 
                     for d, t in enumerate(tokens)]
            item2tokens[item] = tuple(tokens)
        
        print(f"[SID_BUILDER] Converted {len(item2tokens)} items")
        return item2tokens
    
    def _create_reverse_mapping(self, item2tokens):
        """创建反向映射"""
        print(f"[SID_BUILDER] Creating reverse mapping...")
        
        tokens2item = {}
        collisions = 0
        
        for item, tokens in item2tokens.items():
            item_id = self.dataset.item2id[item]
            if tokens in tokens2item:
                collisions += 1
                self.stats['collision_stats'][tokens] += 1
            tokens2item[tokens] = item_id
        
        print(f"[SID_BUILDER] Reverse mapping created with {collisions} collisions")
        return tokens2item
    
    def _save_mappings(self, item2tokens, tokens2item):
        """保存映射文件"""
        print(f"[SID_BUILDER] Saving mapping files...")
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 保存正排索引：item_id → SID-tokens
        item_id2tokens = np.zeros((self.dataset.n_items, self.n_digit), dtype=np.int64)
        for item, tokens in item2tokens.items():
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = np.array(tokens)
        
        np.save(self.item_id2tokens_path, item_id2tokens)
        print(f"[SID_BUILDER] Saved item_id2tokens to {self.item_id2tokens_path}")
        
        # 保存倒排索引：SID-tokens → item_id
        with open(self.tokens2item_path, 'wb') as f:
            pickle.dump(tokens2item, f)
        print(f"[SID_BUILDER] Saved tokens2item to {self.tokens2item_path}")
    
    def _analyze_distribution(self, item2tokens, tokens2item):
        """分析分布"""
        print(f"\n[SID_BUILDER] ===== DISTRIBUTION ANALYSIS =====")
        
        # 基本统计
        self.stats['total_items'] = len(item2tokens)
        self.stats['unique_sids'] = len(tokens2item)
        
        # 统计合法商品
        legal_count = 0
        for item in item2tokens:
            if item in self.dataset.item2id:
                legal_count += 1
        self.stats['legal_items'] = legal_count
        
        print(f"[SID_BUILDER] Total items: {self.stats['total_items']}")
        print(f"[SID_BUILDER] Legal items: {self.stats['legal_items']}")
        print(f"[SID_BUILDER] Unique SIDs: {self.stats['unique_sids']}")
        print(f"[SID_BUILDER] Legal ratio: {self.stats['legal_items']/self.stats['total_items']:.3f}")
        print(f"[SID_BUILDER] Unique ratio: {self.stats['unique_sids']/self.stats['total_items']:.3f}")
        
        # SID分布分析
        sid_counter = Counter()
        digit_counters = [Counter() for _ in range(self.n_digit)]
        
        for item, tokens in item2tokens.items():
            sid_counter[tokens] += 1
            for d, token in enumerate(tokens):
                digit_counters[d][token] += 1
        
        print(f"\n[SID_BUILDER] SID Distribution:")
        print(f"[SID_BUILDER] - Most common SIDs: {sid_counter.most_common(5)}")
        print(f"[SID_BUILDER] - SID collision rate: {(self.stats['total_items'] - self.stats['unique_sids'])/self.stats['total_items']:.3f}")
        
        # 每个digit的分布
        for d in range(self.n_digit):
            counter = digit_counters[d]
            print(f"\n[SID_BUILDER] Digit {d} Distribution:")
            print(f"[SID_BUILDER] - Unique values: {len(counter)}")
            print(f"[SID_BUILDER] - Value range: {min(counter.keys())} - {max(counter.keys())}")
            print(f"[SID_BUILDER] - Most common: {counter.most_common(3)}")
        
        # 保存统计信息
        stats_path = os.path.join(self.cache_dir, f'sid_stats{self.suffix}.json')
        with open(stats_path, 'w') as f:
            json.dump(self._to_jsonable(self.stats), f, indent=2)
        print(f"\n[SID_BUILDER] Saved statistics to {stats_path}")
    
    def _to_jsonable(self, obj):
        """
        把 dict / list / tuple / numpy.* 递归地转换成
        json 可序列化的格式（所有键都是 str，所有值都是
        基础 Python 标量或可序列化容器）
        """
        import numpy as np

        if isinstance(obj, dict):
            # 先把 defaultdict 变成普通 dict，再递归
            return {str(k): self._to_jsonable(v) for k, v in dict(obj).items()}

        elif isinstance(obj, (list, tuple, set)):
            return [self._to_jsonable(v) for v in obj]

        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)

        else:          # str、int、float、bool、None
            return obj
    
    def _check_reconstruction_quality(self, index, sent_embs, item_mask, tag="val"):
        """检查重建质量"""
        print(f"\n[SID_BUILDER] ===== RECONSTRUCTION QUALITY CHECK ({tag.upper()}) =====")
        
        if not np.any(item_mask):
            print(f"[SID_BUILDER] No {tag} items found, skipping quality check")
            return
        
        # 获取被掩码的商品嵌入
        masked_embs = sent_embs[item_mask]
        print(f"[SID_BUILDER] {tag} items count: {len(masked_embs)}")
        
        # 使用索引重建 - 修复：重建指定的验证/测试样本
        idx = np.where(item_mask)[0].astype(np.int64)
        reconstructed = np.vstack([index.reconstruct(int(i)) for i in idx])
        
        # 计算余弦相似度和MSE
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics import mean_squared_error
        
        # 1) 128维评估（已有）
        cos_sim_128 = cosine_similarity(masked_embs, reconstructed).diagonal()
        avg_cos_sim_128 = np.mean(cos_sim_128)
        min_cos_sim_128 = np.min(cos_sim_128)
        max_cos_sim_128 = np.max(cos_sim_128)
        mse_128 = mean_squared_error(masked_embs, reconstructed)
        
        # 2) 新增：映射回768维后再评估
        if hasattr(self, "pca") and self.pca is not None:
            recon_768 = self.pca.inverse_transform(reconstructed)
            orig_768 = self.orig_sent_embs[item_mask]
            cos_sim_768 = cosine_similarity(orig_768, recon_768).diagonal()
            avg_cos_sim_768 = np.mean(cos_sim_768)
            min_cos_sim_768 = np.min(cos_sim_768)
            max_cos_sim_768 = np.max(cos_sim_768)
            mse_768 = mean_squared_error(orig_768, recon_768)
        else:
            cos_sim_768 = avg_cos_sim_768 = min_cos_sim_768 = max_cos_sim_768 = mse_768 = None
        
        print(f"[SID_BUILDER] {tag.upper()} 128-D Reconstruction Quality:")
        print(f"[SID_BUILDER] - Average Cosine Similarity: {avg_cos_sim_128:.4f}")
        print(f"[SID_BUILDER] - Min Cosine Similarity: {min_cos_sim_128:.4f}")
        print(f"[SID_BUILDER] - Max Cosine Similarity: {max_cos_sim_128:.4f}")
        print(f"[SID_BUILDER] - Mean Squared Error: {mse_128:.6f}")
        
        if cos_sim_768 is not None:
            print(f"[SID_BUILDER] {tag.upper()} 768-D Reconstruction Quality:")
            print(f"[SID_BUILDER] - Average Cosine Similarity: {avg_cos_sim_768:.4f}")
            print(f"[SID_BUILDER] - Min Cosine Similarity: {min_cos_sim_768:.4f}")
            print(f"[SID_BUILDER] - Max Cosine Similarity: {max_cos_sim_768:.4f}")
            print(f"[SID_BUILDER] - Mean Squared Error: {mse_768:.6f}")
        
        # 保存质量统计
        quality_stats = {
            'tag': tag,
            'item_count': len(masked_embs),
            'cos128_avg': float(avg_cos_sim_128),
            'cos128_min': float(min_cos_sim_128),
            'cos128_max': float(max_cos_sim_128),
            'mse128': float(mse_128),
            # 新增768维字段
            'cos768_avg': float(avg_cos_sim_768) if cos_sim_768 is not None else None,
            'cos768_min': float(min_cos_sim_768) if cos_sim_768 is not None else None,
            'cos768_max': float(max_cos_sim_768) if cos_sim_768 is not None else None,
            'mse768': float(mse_768) if cos_sim_768 is not None else None,
        }
        
        # 将质量统计添加到总体统计中
        if 'quality_stats' not in self.stats:
            self.stats['quality_stats'] = {}
        self.stats['quality_stats'][tag] = quality_stats
    
    def build(self):
        """构建SID映射"""
        print(f"[SID_BUILDER] ===== STARTING SID BUILDING =====")
        print(f"[SID_BUILDER] Dataset: {self.dataset.__class__.__name__}")
        print(f"[SID_BUILDER] N-digit: {self.n_digit}")
        print(f"[SID_BUILDER] Codebook size: {self.codebook_size}")
        print(f"[SID_BUILDER] Index factory: {self.index_factory}")
        print(f"[SID_BUILDER] Force regenerate: {self.force_regenerate}")
        
        # 检查是否已存在且不强制重新生成
        if not self.force_regenerate and os.path.exists(self.item_id2tokens_path) and os.path.exists(self.tokens2item_path):
            print(f"[SID_BUILDER] Mapping files already exist, loading...")
            self._load_existing_mappings()
            return
        
        # 步骤1: 编码句子嵌入
        if os.path.exists(self.sent_emb_path) and not self.force_regenerate:
            print(f"[SID_BUILDER] Loading existing sentence embeddings...")
            sent_embs = np.fromfile(self.sent_emb_path, dtype=np.float32)
            expected_dim = self.config['sent_emb_dim']
            if sent_embs.size % expected_dim != 0:
                raise ValueError(
                    f"Sent-emb file shape mismatch: total {sent_embs.size} "
                    f"elements, but sent_emb_dim={expected_dim}"
                )
            sent_embs = sent_embs.reshape(-1, expected_dim)
        else:
            sent_embs = self._encode_sent_emb()
        
        # 保存原始768维嵌入用于对比
        self.orig_sent_embs = sent_embs.copy()
        
        # PCA处理
        if self.config['sent_emb_pca'] > 0:
            print(f"[SID_BUILDER] Applying PCA to sentence embeddings...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
            sent_embs = pca.fit_transform(sent_embs)
            self.pca = pca  # 保存PCA对象用于逆变换
            print(f"[SID_BUILDER] PCA completed, new shape: {sent_embs.shape}")
        else:
            self.pca = None
        
        # 步骤2: 获取训练商品
        training_item_mask = self._get_items_for_training()
        
        # 步骤3: 生成语义ID
        if os.path.exists(self.sem_ids_path) and not self.force_regenerate and not self.force_retrain_opq:
            print(f"[SID_BUILDER] Loading existing semantic IDs...")
            with open(self.sem_ids_path, 'r') as f:
                item2sem_ids = json.load(f)
        else:
            print(f"[SID_BUILDER] Generating semantic IDs using OPQ...")
            item2sem_ids = self._generate_semantic_id_opq(sent_embs, training_item_mask)
        
        # 步骤4: 转换为token
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)
        
        # 步骤5: 创建反向映射
        tokens2item = self._create_reverse_mapping(item2tokens)
        
        # 步骤6: 保存映射文件
        self._save_mappings(item2tokens, tokens2item)
        
        # 步骤7: 分析分布
        self._analyze_distribution(item2tokens, tokens2item)
        
        print(f"\n[SID_BUILDER] ===== SID BUILDING COMPLETED =====")
        print(f"[SID_BUILDER] Files generated:")
        print(f"[SID_BUILDER] - {self.item_id2tokens_path}")
        print(f"[SID_BUILDER] - {self.tokens2item_path}")
        print(f"[SID_BUILDER] - {self.sem_ids_path}")
    
    def _load_existing_mappings(self):
        """加载已存在的映射文件"""
        print(f"[SID_BUILDER] Loading existing mappings...")
        
        # 加载正排索引
        item_id2tokens = np.load(self.item_id2tokens_path)
        print(f"[SID_BUILDER] Loaded item_id2tokens shape: {item_id2tokens.shape}")
        
        # 加载倒排索引
        with open(self.tokens2item_path, 'rb') as f:
            tokens2item = pickle.load(f)
        print(f"[SID_BUILDER] Loaded tokens2item with {len(tokens2item)} entries")
        
        # 重建item2tokens
        item2tokens = {}
        for item_id in range(1, self.dataset.n_items):
            item = self.dataset.id_mapping['id2item'][item_id]
            tokens = tuple(item_id2tokens[item_id])
            if not np.all(tokens == 0):  # 跳过全0的PAD
                item2tokens[item] = tokens
        
        # 分析分布
        self._analyze_distribution(item2tokens, tokens2item)


def main():
    parser = argparse.ArgumentParser(description="Standalone SID Builder for DIFF_GRM")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", default="DIFF_GRM", help="Model name")
    parser.add_argument("--n_digit", type=int, default=4, help="Number of digits")
    parser.add_argument("--codebook_size", type=int, default=256, help="Codebook size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Force regenerate all files")
    parser.add_argument("--disable_opq", action="store_true", help="Use plain PQ instead of OPQ")
    parser.add_argument("--sent_emb_model", default="sentence-transformers/all-MiniLM-L6-v2", 
                       help="Sentence embedding model")
    parser.add_argument("--sent_emb_batch_size", type=int, default=32, help="Sentence embedding batch size")
    parser.add_argument("--sent_emb_pca", type=int, default=0, help="PCA dimensions (0 for no PCA)")
    parser.add_argument("--embed_use_gpu", action="store_true", help="Encode sentence embeddings on GPU")
    parser.add_argument("--opq_use_gpu", action="store_true", help="Use GPU for OPQ")
    parser.add_argument("--opq_gpu_id", type=int, default=0, help="GPU ID for OPQ")
    parser.add_argument("--category", default=None, help="Item category (for Amazon datasets)")
    parser.add_argument("--sent_emb_dim", type=int, default=768, help="Original sentence embedding dimension")
    parser.add_argument("--faiss_omp_num_threads", type=int, default=4, help="FAISS threads")
    args = parser.parse_args()
    
    # 初始化
    init_seed(args.seed, True)
    device, use_ddp = init_device()
    
    # 获取配置
    config = get_config(
        model_name=args.model,
        dataset_name=args.dataset,
        config_file=None,
        config_dict=None
    )
    config.update({
        "n_digit": args.n_digit,
        "codebook_size": args.codebook_size,
        "category": args.category,
        "sent_emb_model": args.sent_emb_model,
        "sent_emb_dim": args.sent_emb_dim,
        "sent_emb_batch_size": args.sent_emb_batch_size,
        "sent_emb_pca": args.sent_emb_pca,
        "embed_use_gpu": args.embed_use_gpu,
        "opq_use_gpu": args.opq_use_gpu,
        "opq_gpu_id": args.opq_gpu_id,
        "faiss_omp_num_threads": args.faiss_omp_num_threads,
        "disable_opq": args.disable_opq,
        "device": device,
        "use_ddp": use_ddp
    })
    
    # 创建Accelerator（和Pipeline里一模一样，只是项目名不同）
    project_dir = os.path.join(
        config.get('tensorboard_log_dir', 'runs'),  # 训练默认配里就有
        config['dataset'],                          # 如 AmazonReviews2014
        config['model'] + '_SID_builder'            # 给 SID builder 单独开一个子目录
    )
    
    config['accelerator'] = Accelerator(
        log_with='tensorboard',     # 训练时就是这样写的
        project_dir=project_dir
    )
    
    # 初始化日志
    init_logger(config)
    
    # 获取数据集
    dataset_cls = get_dataset(args.dataset)
    dataset = dataset_cls(config)
    
    # 构建SID
    builder = StandaloneSIDBuilder(config, dataset, force_regenerate=args.force)
    builder.build()


if __name__ == "__main__":
    main() 