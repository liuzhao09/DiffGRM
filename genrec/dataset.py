# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
from datasets import Dataset


class AbstractDataset:
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = self.config['accelerator']
        self.logger = getLogger()

        self.all_item_seqs = {}
        self.id_mapping = {
            'user2id': {'[PAD]': 0},
            'item2id': {'[PAD]': 0},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = None
        self.split_data = None

    def __str__(self) -> str:
        return f'[Dataset] {self.__class__.__name__}\n' \
                f'\tNumber of users: {self.n_users}\n' \
                f'\tNumber of items: {self.n_items}\n' \
                f'\tNumber of interactions: {self.n_interactions}\n' \
                f'\tAverage item sequence length: {self.avg_item_seq_len}'

    @property
    def n_users(self):
        """
        Returns the number of users in the dataset.

        Returns:
            int: The number of users in the dataset.
        """
        return len(self.user2id)

    @property
    def n_items(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.item2id)

    @property
    def n_interactions(self):
        """
        Returns the total number of interactions in the dataset.

        Returns:
            int: The total number of interactions.
        """
        n_inters = 0
        for user in self.all_item_seqs:
            n_inters += len(self.all_item_seqs[user])
        return n_inters

    @property
    def avg_item_seq_len(self):
        """
        Returns the average length of item sequences in the dataset.

        Returns:
            float: The average length of item sequences.
        """
        return self.n_interactions / self.n_users

    @property
    def user2id(self):
        """
        Returns the user-to-id mapping.

        Returns:
            dict: The user-to-id mapping.
        """
        return self.id_mapping['user2id']

    @property
    def item2id(self):
        """
        Returns the item-to-id mapping.

        Returns:
            dict: The item-to-id mapping.
        """
        return self.id_mapping['item2id']

    def _download_and_process_raw(self):
        """
        This method should be implemented in the subclass.
        It is responsible for downloading and processing the raw data.
        """
        raise NotImplementedError('This method should be implemented in the subclass')

    def _leave_one_out(self):
        """
        Splits the dataset into train, validation, and test sets using the leave-one-out strategy.

        Returns:
            dict: A dictionary containing the train, validation, and test datasets.
                  Each dataset is represented as a dictionary with 'user' and 'item_seq' keys.
                  The 'user' key contains a list of users, and the 'item_seq' key contains a list of item sequences.
        """
        datasets = {'train': {'user': [], 'item_seq': []},
                    'val': {'user': [], 'item_seq': []},
                    'test': {'user': [], 'item_seq': []}}
        for user in self.all_item_seqs:
            datasets['test']['user'].append(user)
            datasets['test']['item_seq'].append(self.all_item_seqs[user])
            if len(self.all_item_seqs[user]) > 1:
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(self.all_item_seqs[user][:-1])
            if len(self.all_item_seqs[user]) > 2:
                datasets['train']['user'].append(user)
                datasets['train']['item_seq'].append(self.all_item_seqs[user][:-2])
        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])
        return datasets

    def _sliding_train(self, min_hist=2, max_hist=50):
        """
        只扩充 train，val / test 仍保持 leave-one-out。
        保证训练样本的 label 不与 val/test 的 target（最后一条交互）重叠
        
        Args:
            min_hist (int): 滑窗最短history长度（含label前的序列长度）
            max_hist (int): 滑窗最长history长度
            
        Returns:
            dict: A dictionary containing the train, validation, and test datasets.
        """
        datasets = {'train': {'user': [], 'item_seq': []},
                    'val': {'user': [], 'item_seq': []},
                    'test': {'user': [], 'item_seq': []}}

        for user, seq in self.all_item_seqs.items():
            L = len(seq)

            # test: 用完整序列
            datasets['test']['user'].append(user)
            datasets['test']['item_seq'].append(seq)

            # val: 去掉最后 1 个
            if L > 1:
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(seq[:-1])

            # train: 滑窗
            if L > 2:  # 至少需要3个item才能生成训练样本
                # 关键修复：hi_max只能到L-3，确保label ≤ L-3
                # val和test的历史都是seq[:-1]（0...L-2）
                # 把训练label最远限制到L-3，就保证label ∉ 这段history
                hi_max = min(L - 3, max_hist)
                
                if hi_max >= min_hist:
                    for t in range(min_hist, hi_max + 1):
                        datasets['train']['user'].append(user)
                        datasets['train']['item_seq'].append(seq[:t + 1])  # 长度 t+1，最后 1 个当 label

        # 转成 HF Dataset
        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])
        return datasets

    def split(self):
        """
        Split the dataset into train, validation, and test sets based on the specified split strategy.

        Returns:
            datasets (dict): A dictionary containing the train and test datasets.
        """
        if self.split_data is not None:
            return self.split_data

        # ① 保留 split 字段作总开关
        strategy = self.config.get('split', 'leave_one_out')
        
        # 改进布尔值解析
        val = self.config.get('train_sliding', False)
        slide = val if isinstance(val, bool) else str(val).lower() == "true"

        if strategy in ['leave_one_out', 'last_out']:
            if slide:  # leave-one-out + train 滑窗
                datasets = self._sliding_train(
                    min_hist=self.config.get('min_hist_len', 2),
                    max_hist=self.config.get('max_hist_len', 50)
                )
            else:  # 纯 leave-one-out
                datasets = self._leave_one_out()
        else:
            raise NotImplementedError(f'Split strategy [{strategy}] not implemented.')

        self.split_data = datasets
        return self.split_data

    def log(self, message, level='info'):
        from genrec.utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)
