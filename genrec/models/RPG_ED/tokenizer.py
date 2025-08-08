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


class RPG_EDTokenizer(AbstractTokenizer):
    """
    RPG_ED Tokenizer for Encoder-Decoder architecture
    
    Special tokens:
    - PAD=0, BOS=1, EOS=2, USR_BUCKET_OFFSET=3, SID_OFFSET=3+n_user_buckets
    
    SID Configuration:
    - n_digit=4, codebook_size=256
    - vocab_size = 3 + n_user_buckets + n_digit * codebook_size
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        self.n_codebook_bits = self._get_codebook_bits(config['codebook_size'])
        self.index_factory = f'OPQ{config["n_digit"]},IVF1,PQ{config["n_digit"]}x{self.n_codebook_bits}'

        super(RPG_EDTokenizer, self).__init__(config, dataset)
        self.dataset = dataset  # 添加dataset引用
        self.item2id = dataset.item2id
        self.id2item = dataset.id_mapping['id2item']
        
        # Special tokens - 简化token ID分配
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.sid_offset = 3  # SID token从3开始
        
        self.item2tokens = self._init_tokenizer(dataset)
        
        # Create reverse mapping for inference
        self.tokens2item = self._create_reverse_mapping()
        
        # Save mappings
        self._save_mappings()
        
        # Set collate functions
        from genrec.models.RPG_ED.collate import collate_fn_train, collate_fn_val, collate_fn_test
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
        """编码句子嵌入"""
        assert self.config['metadata'] == 'sentence', \
            'RPG_EDTokenizer only supports sentence metadata.'

        meta_sentences = []
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        if 'sentence-transformers' in self.config['sent_emb_model']:
            sent_emb_model = SentenceTransformer(
                self.config['sent_emb_model']
            ).to(self.config['device'])

            sent_embs = sent_emb_model.encode(
                meta_sentences,
                convert_to_numpy=True,
                batch_size=self.config['sent_emb_batch_size'],
                show_progress_bar=True,
                device=self.config['device']
            )
        else:
            raise ValueError(f"Unsupported embedding model: {self.config['sent_emb_model']}")

        sent_embs.tofile(output_path)
        return sent_embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """获取训练用的商品"""
        items_for_training = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        
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
        """使用OPQ生成语义ID"""
        import faiss
        
        # 添加调试信息
        self.log(f'[TOKENIZER] sent_embs shape: {sent_embs.shape}')
        self.log(f'[TOKENIZER] train_mask shape: {train_mask.shape}')
        self.log(f'[TOKENIZER] train_mask True count: {np.sum(train_mask)}')
        
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

        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for i in range(self.n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)
        pq_codes = np.array(faiss_sem_ids)

        item2sem_ids = {}
        for i in range(pq_codes.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(pq_codes[i].tolist())
        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
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
        # 加载语义ID
        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_{self.index_factory}.sem_ids'
        )

        if not os.path.exists(sem_ids_path):
            # 加载或编码句子嵌入
            sent_emb_path = os.path.join(
                dataset.cache_dir, 'processed',
                f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
            )
            if os.path.exists(sent_emb_path):
                self.log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
                sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings...')
                sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
            
            # PCA
            if self.config['sent_emb_pca'] > 0:
                self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
            self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

            # 生成语义ID
            training_item_mask = self._get_items_for_training(dataset)
            
            # 添加调试信息
            self.log(f'[TOKENIZER] sent_embs shape: {sent_embs.shape}')
            self.log(f'[TOKENIZER] training_item_mask shape: {training_item_mask.shape}')
            self.log(f'[TOKENIZER] training_item_mask True count: {np.sum(training_item_mask)}')
            
            self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

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
        cache_dir = os.path.join(self.dataset.cache_dir, 'processed')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 保存正排索引
        item_id2tokens = np.zeros((self.dataset.n_items, self.n_digit), dtype=np.int64)
        for item, tokens in self.item2tokens.items():
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = np.array(tokens)
        
        np.save(os.path.join(cache_dir, 'item_id2tokens.npy'), item_id2tokens)
        
        # 保存倒排索引
        with open(os.path.join(cache_dir, 'tokens2item.pkl'), 'wb') as f:
            pickle.dump(self.tokens2item, f)
        
        self.log(f'[TOKENIZER] Saved mappings to {cache_dir}')

    def encode_history(self, item_seq, max_len=50):
        """编码用户历史序列"""
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
        """编码decoder输入 - 修复为正确的teacher forcing"""
        if target_item in self.item2tokens:
            tokens = list(self.item2tokens[target_item])  # 4个token ID（带offset）
            
            # 将token ID转换为codebook ID
            codebook_tokens = []
            for digit, token_id in enumerate(tokens):
                codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                codebook_tokens.append(codebook_id)
            
            # 修复：正确的teacher forcing
            # decoder输入：[BOS, sid0, sid1, sid2]（前3个真实token + BOS）
            # decoder标签：[sid0, sid1, sid2, sid3]（4个目标token）
            
            # 构建用于embedding lookup的输入token（需要转换回token ID）
            input_tokens = [self.bos_token]  # 从BOS开始
            for i in range(3):  # 只取前3个codebook token
                # 将codebook ID转换回token ID用于embedding
                token_id = codebook_tokens[i] + (self.sid_offset + i * self.codebook_size)
                input_tokens.append(token_id)
            
            decoder_input = input_tokens  # [BOS, token0, token1, token2]
            decoder_labels = codebook_tokens  # [cb0, cb1, cb2, cb3] - 4个codebook ID
        else:
            # 未知商品
            decoder_input = [self.bos_token] + [self.pad_token] * 3  # 长度4
            decoder_labels = [self.pad_token] * 4  # 长度4
        
        return decoder_input, decoder_labels  # decoder_input长度4, decoder_labels长度4

    def decode_tokens_to_item(self, tokens):
        """将token序列解码为商品ID"""
        if len(tokens) != self.n_digit:
            return None
        
        token_tuple = tuple(tokens)
        return self.tokens2item.get(token_tuple)

    def tokenize_function(self, example: dict, split: str) -> dict:
        """tokenize函数"""
        item_seq = example['item_seq']  # Python list
        target_item = item_seq[-1]  # 原始字符串
        target_item_id = self.item2id.get(target_item, 0)  # 转换为整数ID，0为未知商品
        
        # 编码历史
        history_sid = self.encode_history(item_seq[:-1] if split == 'train' else item_seq)
        
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