# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class AR_GRMTokenizer(AbstractTokenizer):
    """
    AR_GRM Tokenizer for autoregressive generative recommendation

    Special tokens:
    - PAD=0, BOS=1, EOS=2, SID_OFFSET=3

    SID Configuration:
    - n_digit: configurable (e.g., 4, 8, 12), codebook_size=256
    - vocab_size = 3 + n_digit * codebook_size
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        # 兜底，避免 KeyError
        config.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        config.setdefault('num_proc', 1)

        self.n_codebook_bits = self._get_codebook_bits(config['codebook_size'])
        
        # 统一量化器：支持 sid_quantizer（兼容 quantizer）
        sid_q = config.get('sid_quantizer', None)
        if sid_q is None:
            q_old = config.get('quantizer', 'pq').lower()
            sid_q = {'pq': 'opq_pq', 'rq': 'rq_kmeans'}.get(q_old, 'opq_pq')
        self.sid_quantizer = sid_q
        assert self.sid_quantizer in ('opq_pq', 'rq_kmeans', 'none'), \
            f"sid_quantizer must be one of ['opq_pq','rq_kmeans','none'], got {self.sid_quantizer}"
        
        # 与 DIFF_GRM 对齐的 index_factory
        if self.sid_quantizer == 'opq_pq':
            use_opq = not config.get('disable_opq', False)
            if use_opq:
                self.index_factory = f'OPQ{config["n_digit"]},IVF1,PQ{config["n_digit"]}x{self.n_codebook_bits}'
            else:
                self.index_factory = f'IVF1,PQ{config["n_digit"]}x{self.n_codebook_bits}'
        elif self.sid_quantizer == 'rq_kmeans':
            self.index_factory = f'RQKMEANS{config["n_digit"]}x{self.n_codebook_bits}'
        else:  # 'none'
            self.index_factory = f'RAND{config["n_digit"]}x{self.n_codebook_bits}'

        # 先初始化父类，保证 self.config / self.logger 等字段可用
        super(AR_GRMTokenizer, self).__init__(config, dataset)
        
        # 现在再写日志
        self.log(f'[TOKENIZER] Index factory: {self.index_factory}')
        self.dataset = dataset  # 添加dataset引用
        self.item2id = dataset.item2id
        self.id2item = dataset.id_mapping['id2item']
        
        # Special tokens - 简化token ID分配
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.mask_token = -1  # MASK token用于推理，不在vocab中
        self.sid_offset = 3  # SID token从3开始
        
        self.item2tokens = self._init_tokenizer(dataset)
        
        # Create reverse mapping for inference (如果还没有创建的话)
        if not hasattr(self, 'tokens2item'):
            self.tokens2item = self._create_reverse_mapping()
        
        # Set collate functions
        # 直接使用本目录下 collate
        from .collate import collate_fn_train, collate_fn_val, collate_fn_test
        self.collate_fn = {
            'train': collate_fn_train,
            'val': collate_fn_val,
            'test': collate_fn_test
        }

    @property
    def n_digit(self):
        return self.config['n_digit']

    @property
    def codebook_size(self):
        return self.config['codebook_size']

    @property
    def max_token_seq_len(self) -> int:
        return 1 + self.n_digit  # [BOS] + n_digit SID tokens

    @property
    def vocab_size(self) -> int:
        return 3 + self.n_digit * self.codebook_size  # PAD(0) + BOS(1) + EOS(2) + SID tokens

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """编码句子嵌入：支持任意 Hugging Face SentenceTransformer 模型 id，并做向量归一化"""
        assert self.config['metadata'] == 'sentence', \
            'AR_GRMTokenizer only supports sentence metadata.'

        meta_sentences = []
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        # 接受任意HF模型id（如 Alibaba-NLP/gte-large-en-v1.5 或 BAAI/bge-large-en-v1.5）
        model_id = self.config['sent_emb_model']
        sent_emb_model = SentenceTransformer(model_id, trust_remote_code=True).to(self.config['device'])

        # 直接encode（GTE/BGE无需前缀），并进行L2归一化
        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device'],
            normalize_embeddings=True,
        )

        # 按模型basename分别落盘，避免不同模型冲突
        sent_embs.tofile(output_path)
        return sent_embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """获取训练用的商品"""
        items_for_training = set()
        
        # 首先触发数据集分割（如果还没有分割）
        split_data = dataset.split()
        
        # 从训练集中收集所有items
        if 'train' in split_data:
            train_dataset = split_data['train']
            # train_dataset是Hugging Face Dataset对象
            if hasattr(train_dataset, 'column_names') and 'item_seq' in train_dataset.column_names:
                # 遍历所有item_seq
                for item_seq in train_dataset['item_seq']:
                    if isinstance(item_seq, (list, tuple)):
                        items_for_training.update(item_seq)
                    else:
                        items_for_training.add(item_seq)
        
        # 修复：确保mask大小与sent_embs匹配
        # sent_embs只包含item_id从1到n_items-1的商品
        n_sent_embs = dataset.n_items - 1  # 与_encode_sent_emb中的range(1, dataset.n_items)匹配
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {n_sent_embs}')
        self.log(f'[TOKENIZER] Training items sample: {list(items_for_training)[:10]}')
        
        mask = np.zeros(n_sent_embs, dtype=bool)
        for item in items_for_training:
            item_id = dataset.item2id[item]
            if 1 <= item_id < dataset.n_items:  # 确保item_id在有效范围内
                mask[item_id - 1] = True  # 转换为0-based索引
        
        self.log(f'[TOKENIZER] Mask shape: {mask.shape}, True count: {np.sum(mask)}')
        return mask

    def _generate_semantic_id_opq(self, sent_embs, sem_ids_path, train_mask):
        """使用OPQ/PQ生成语义ID（兼容 disable_opq），并用 invlists 的 ids 对齐。"""
        import faiss
        
        # 调试信息
        self.log(f'[TOKENIZER] sent_embs shape: {sent_embs.shape}')
        self.log(f'[TOKENIZER] train_mask shape: {train_mask.shape}')
        self.log(f'[TOKENIZER] train_mask True count: {np.sum(train_mask)}')

        # 构建索引
        if self.config['opq_use_gpu']:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log(f'[TOKENIZER] Training index...')
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)

        # 兼容 IndexPreTransform 与非 PreTransform
        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
        else:
            ivf_index = faiss.downcast_index(index)

        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        # 取 codes 与 ids，并保持同序对齐
        codes_ptr = invlists.get_codes(0)
        ids_ptr = invlists.get_ids(0)
        pq_codes_u8 = faiss.rev_swig_ptr(codes_ptr, ls * invlists.code_size)
        ids = faiss.rev_swig_ptr(ids_ptr, ls).copy()
        pq_codes_u8 = pq_codes_u8.reshape(-1, invlists.code_size)

        # 解析 PQ Code
        faiss_sem_ids = []
        n_bytes = invlists.code_size
        for u8code in pq_codes_u8:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for _ in range(self.n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)

        # 用 ids 对齐 item 顺序
        item2sem_ids = {}
        for pos, iid0 in enumerate(ids):
            item = self.id2item[int(iid0) + 1]
            item2sem_ids[item] = tuple(int(v) for v in faiss_sem_ids[pos])

        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _generate_semantic_id_rq_kmeans(self, sent_embs, sem_ids_path, train_mask):
        """使用 Residual Quantization（KMeans）生成语义ID（与 DIFF_GRM 对齐）"""
        import faiss
        d = sent_embs.shape[1]
        K = self.codebook_size
        niter = int(self.config.get('rq_kmeans_niters', 20))
        seed = int(self.config.get('rq_kmeans_seed', 1234))
        
        # 初始化残差为原始向量
        residuals = sent_embs.copy().astype(np.float32, copy=False)
        codes_all = np.zeros((sent_embs.shape[0], self.n_digit), dtype=np.int64)
        
        for stage in range(self.n_digit):
            kmeans = faiss.Kmeans(d=d, k=K, niter=niter, verbose=False, seed=seed + stage)
            kmeans.train(residuals[train_mask])
            # In current Faiss Python, Kmeans.centroids is already a numpy array
            centroids = np.asarray(kmeans.centroids, dtype=np.float32)
            if centroids.ndim == 1:
                centroids = centroids.reshape(K, d)
            elif centroids.shape == (d, K):
                centroids = centroids.T
            assert centroids.shape == (K, d), f"centroids shape {centroids.shape} != {(K, d)}"
            
            # 为全部样本分配最近质心
            index = faiss.IndexFlatL2(d)
            index.add(centroids)
            D, I = index.search(residuals, 1)  # I: [N, 1]
            codes_all[:, stage] = I[:, 0].astype(np.int64)
            
            # 更新残差
            residuals = residuals - centroids[I[:, 0]]
        
        # 转成 dict
        item2sem_ids = {}
        for i in range(codes_all.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(int(v) for v in codes_all[i].tolist())
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _generate_semantic_id_random(self, sem_ids_path, n_items, seed=12345):
        """为每个商品随机生成 n_digit 个 codebook ID（均匀[0, K-1]）"""
        rng = np.random.default_rng(seed)
        item2sem_ids = {}
        for i in range(1, n_items):
            item = self.id2item[i]
            codes = rng.integers(low=0, high=self.codebook_size, size=self.n_digit, endpoint=False, dtype=np.int64)
            item2sem_ids[item] = tuple(int(c) for c in codes.tolist())
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """将语义ID转换为token"""
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            # 修复：重新引入offset，避免与PAD/BOS冲突
            # 每个digit的codebook ID加上对应的offset
            tokens = [t + self.sid_offset + d * self.codebook_size 
                     for d, t in enumerate(tokens)]
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """初始化tokenizer"""
        # 构建路径（与 DIFF 一致）
        dataset_name = dataset.__class__.__name__
        if hasattr(dataset, 'category') and dataset.category:
            cache_dir = os.path.join(dataset.cache_dir, 'processed')
        else:
            cache_dir = os.path.join('data', dataset_name, 'processed')
        os.makedirs(cache_dir, exist_ok=True)

        # 量化器标签（含 seed/iters）
        model_basename = os.path.basename(self.config["sent_emb_model"])
        quant_tag = self.index_factory
        if self.sid_quantizer == 'rq_kmeans':
            quant_tag += f'_seed{self.config.get("rq_kmeans_seed",1234)}_it{self.config.get("rq_kmeans_niters",20)}'
        elif self.sid_quantizer == 'none':
            quant_tag += f'_seed{self.config.get("sid_random_seed",12345)}'

        sem_ids_path = os.path.join(
            cache_dir,
            f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}.sem_ids'
        )

        # 两份嵌入文件：raw / pca
        raw_path = os.path.join(cache_dir, f'{model_basename}_raw_d{self.config["sent_emb_dim"]}.sent_emb')
        pca_path = os.path.join(cache_dir, f'{model_basename}_pca{self.config["sent_emb_pca"]}.sent_emb')

        force_regenerate = self.config.get('force_regenerate_opq', False) or self.config.get('force_regenerate_codes', False)

        # 按量化器选择向量管线
        sent_embs = None
        if self.sid_quantizer == 'opq_pq':
            if self.config['sent_emb_pca'] > 0 and os.path.exists(pca_path):
                self.log(f'[TOKENIZER] Loading PCA-ed sentence embeddings from {pca_path}...')
                sent_embs = np.fromfile(pca_path, dtype=np.float32).reshape(-1, self.config['sent_emb_pca'])
            elif os.path.exists(raw_path):
                self.log(f'[TOKENIZER] Loading RAW sentence embeddings from {raw_path}...')
                raw_embs = np.fromfile(raw_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
                if self.config['sent_emb_pca'] > 0:
                    self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                    training_item_mask = self._get_items_for_training(dataset)
                    pca.fit(raw_embs[training_item_mask])
                    sent_embs = pca.transform(raw_embs).astype(np.float32, copy=False)
                    if self.config.get('normalize_after_pca', True):
                        norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
                        sent_embs = sent_embs / norms
                    sent_embs.tofile(pca_path)
                else:
                    sent_embs = raw_embs
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings...')
                raw_embs = self._encode_sent_emb(dataset, raw_path)
                if self.config['sent_emb_pca'] > 0:
                    self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                    training_item_mask = self._get_items_for_training(dataset)
                    pca.fit(raw_embs[training_item_mask])
                    sent_embs = pca.transform(raw_embs).astype(np.float32, copy=False)
                    if self.config.get('normalize_after_pca', True):
                        norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
                        sent_embs = sent_embs / norms
                    sent_embs.tofile(pca_path)
                else:
                    sent_embs = raw_embs
            self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

        elif self.sid_quantizer == 'rq_kmeans':
            # RQ-KMeans：只用 RAW，不做 PCA
            if os.path.exists(raw_path):
                self.log(f'[TOKENIZER] Loading RAW sentence embeddings from {raw_path}...')
                sent_embs = np.fromfile(raw_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings (RAW, no PCA for RQ-KMeans)...')
                sent_embs = self._encode_sent_emb(dataset, raw_path)
            self.log(f'[TOKENIZER] Sentence embeddings shape (RAW): {sent_embs.shape}')

        # 生成或加载量化结果
        if force_regenerate or not os.path.exists(sem_ids_path):
            if force_regenerate:
                self.log(f'[TOKENIZER] Force regenerating quantization results ({self.sid_quantizer})...')
            else:
                self.log(f'[TOKENIZER] Quantization results not found, generating ({self.sid_quantizer})...')
            training_item_mask = self._get_items_for_training(dataset)
            if self.sid_quantizer == 'opq_pq':
                self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)
            elif self.sid_quantizer == 'rq_kmeans':
                self._generate_semantic_id_rq_kmeans(sent_embs, sem_ids_path, training_item_mask)
            else:  # 'none'
                self._generate_semantic_id_random(
                    sem_ids_path, n_items=self.dataset.n_items,
                    seed=int(self.config.get('sid_random_seed', 12345))
                )
        else:
            self.log(f'[TOKENIZER] Using existing quantization results from {sem_ids_path}')

        # 加载 sem_ids → tokens
        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        # 同步映射文件名（含 seed/iters）
        map_tag = f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}_{self.n_digit}d'
        fwd_path = os.path.join(cache_dir, f'item_id2tokens_{map_tag}.npy')
        inv_path = os.path.join(cache_dir, f'tokens2item_{map_tag}.pkl')

        if force_regenerate:
            fwd_exists = inv_exists = False
            self.log(f'[TOKENIZER] Force regenerate enabled, ignoring existing mapping files')
        else:
            fwd_exists = os.path.exists(fwd_path)
            inv_exists = os.path.exists(inv_path)

        if fwd_exists and inv_exists:
            self.log(f'[TOKENIZER] Loading existing mappings for tag: {map_tag} from {fwd_path}')
            item_id2tokens = np.load(fwd_path)
            item2tokens = {}
            for iid, toks in enumerate(item_id2tokens):
                if iid == 0:
                    continue
                item2tokens[self.id2item[iid]] = tuple(toks.tolist())
            with open(inv_path, 'rb') as f:
                self.tokens2item = pickle.load(f)
            self.log(f'[TOKENIZER] Successfully loaded {len(item2tokens)} item mappings')
        else:
            self.item2tokens = item2tokens
            self.tokens2item = self._create_reverse_mapping()
            self._save_mappings()

        if not hasattr(self, 'item2tokens'):
            self.item2tokens = item2tokens
        return item2tokens

    def _create_reverse_mapping(self):
        """创建反向映射用于推理"""
        tokens2item = {}
        for item, tokens in self.item2tokens.items():
            item_id = self.dataset.item2id[item]
            tokens2item[tuple(tokens)] = item_id
        return tokens2item

    def _save_mappings(self):
        """保存映射文件"""
        # 构建路径（与 DIFF 一致）
        dataset_name = self.dataset.__class__.__name__
        if hasattr(self.dataset, 'category') and self.dataset.category:
            cache_dir = os.path.join(self.dataset.cache_dir, 'processed')
        else:
            cache_dir = os.path.join('data', dataset_name, 'processed')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 文件名包含：模型+PCA+量化器标签(+种子/iters)+n_digit，完全避免不同配置冲突
        model_basename = os.path.basename(self.config["sent_emb_model"]) 
        quant_tag = self.index_factory
        if self.sid_quantizer == 'rq_kmeans':
            quant_tag += f'_seed{self.config.get("rq_kmeans_seed",1234)}_it{self.config.get("rq_kmeans_niters",20)}'
        elif self.sid_quantizer == 'none':
            quant_tag += f'_seed{self.config.get("sid_random_seed",12345)}'
        map_tag = f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}_{self.n_digit}d'
        
        # 保存正排索引：item_id → SID-tokens
        item_id2tokens = np.zeros((self.dataset.n_items, self.n_digit), dtype=np.int64)
        for item, tokens in self.item2tokens.items():
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = np.array(tokens)
        
        np.save(os.path.join(cache_dir, f'item_id2tokens_{map_tag}.npy'), item_id2tokens)
        
        # 保存倒排索引：SID-tokens → item_id
        with open(os.path.join(cache_dir, f'tokens2item_{map_tag}.pkl'), 'wb') as f:
            pickle.dump(self.tokens2item, f)
        
        self.log(f'[TOKENIZER] Saved mappings with tag: {map_tag} to {cache_dir}')
        self.log(f'[TOKENIZER] Files: item_id2tokens_{map_tag}.npy, tokens2item_{map_tag}.pkl')

    def encode_history(self, item_seq, max_len=None):
        """编码用户历史序列"""
        if max_len is None:
            max_len = self.config.get('max_history_len', 50)
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]
        
        history_sid = []
        for item in item_seq:
            if item in self.item2tokens:
                history_sid.append(list(self.item2tokens[item]))
            else:
                # 未知商品用PAD填充
                history_sid.append([self.pad_token] * self.n_digit)
        
        # 填充到固定长度
        while len(history_sid) < max_len:
            history_sid.append([self.pad_token] * self.n_digit)
        
        return history_sid  # 返回list，让datasets.map自动张量化

    def encode_decoder_input(self, target_item):
        """编码decoder输入 - 与RPG_ED保持一致"""
        if target_item in self.item2tokens:
            tokens = list(self.item2tokens[target_item])  # 4个token ID（带offset）
            
            # 将token ID转换为codebook ID
            codebook_tokens = []
            for digit, token_id in enumerate(tokens):
                codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                codebook_tokens.append(codebook_id)
            
            # decoder输入和标签都是codebook IDs
            decoder_input = codebook_tokens  # [cb0, cb1, cb2, cb3]
            decoder_labels = codebook_tokens  # [cb0, cb1, cb2, cb3]
        else:
            # 🚀 修复：未知商品使用-100作为ignore_index，避免与合法codebook 0冲突
            decoder_input = [self.pad_token] * self.n_digit  # 长度n_digit
            decoder_labels = [-100] * self.n_digit  # 使用-100作为ignore_index
        
        return decoder_input, decoder_labels

    def decode_tokens_to_item(self, tokens):
        """将token序列解码为商品ID"""
        if len(tokens) != self.n_digit:
            return None
        
        token_tuple = tuple(tokens)
        return self.tokens2item.get(token_tuple)

    def codebooks_to_item_id(self, cb_ids):
        """
        将codebook ID序列转换为item_id，检查合法性
        
        Args:
            cb_ids: List[int] 长度 n_digit, 原始 codebook ID (0-255)
            
        Returns:
            item_id(int) 或 None（如果非法）
        """
        if len(cb_ids) != self.n_digit:
            return None
        
        # 将codebook ID转换为token ID
        token_ids = [
            cb_ids[d] + self.sid_offset + d * self.codebook_size
            for d in range(self.n_digit)
        ]
        
        # 查找对应的item_id
        return self.tokens2item.get(tuple(token_ids))

    def tokenize_function(self, example: dict, split: str) -> dict:
        """tokenize函数 - 修复数据泄露问题"""
        item_seq = example['item_seq']  # Python list
        target_item = item_seq[-1]  # 原始字符串
        
        # 修复：所有split都应该用item_seq[:-1]作为历史，避免数据泄露
        history_sid = self.encode_history(item_seq[:-1])
        
        if split == 'train':
            # 训练时编码decoder输入
            decoder_input, decoder_labels = self.encode_decoder_input(target_item)
            return {
                'history_sid': history_sid,  # 直接list
                'decoder_input_ids': decoder_input,  # 直接list
                'decoder_labels': decoder_labels  # 直接list
            }
        else:
            # 验证/测试时生成真标签
            _, decoder_labels = self.encode_decoder_input(target_item)
            return {
                'history_sid': history_sid,  # 直接list
                'labels': decoder_labels  # 新增：真标签序列
            }

    def tokenize(self, datasets: dict) -> dict:
        """tokenize数据集"""
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=False,  # 关闭批处理，避免数据结构混乱
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets 