import os.path as osp
import pickle

import torch.utils.data
from icecream import ic
from torch.utils.data import Dataset


class ESNLIDataset(Dataset):
    """
    A dataset for ESNLI
    """

    def __init__(self, path: str, split: str = 'train', frac=1.0, chunk=0):
        assert split in {'train', 'val', 'test'}, 'split must be one of train, val, test'
        assert 0.0 <= frac <= 1.0, 'frac must be between 0 and 1'

        super().__init__()
        self.name = f'ESNLI_{frac}'

        # Load pickle file
        esnli_path = osp.join(path, f'esnli_{frac}')
        keys = ['Sentences', 'Sentences_embedding', 'concept_ids', 'gold_label', 'Explanation_1'] # 'Explanation_2', 'Explanation_3', 'gold_label',
        string_keys = ['Sentences', 'Explanation_1']

        self.esnli = {}
        for k in keys :
            key_path = osp.join(esnli_path, f'{split}_{k}', f'chunk{chunk}.pkl')
            try:
                self.esnli[k] = pickle.load(open(key_path, 'rb'))
                if isinstance(self.esnli[k], torch.Tensor):
                    self.esnli[k] = self.esnli[k]
            except Exception as e:
                ic(e, key_path)

        for k in self.esnli:
            if k in string_keys:
                self.esnli[k] = [str(elt) for elt in self.esnli[k]]

    def __len__(self):
        return len(self.esnli['gold_label'])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.esnli.items()}
