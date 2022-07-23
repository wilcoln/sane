import torch.utils.data
import pandas as pd
import os.path as osp
from tqdm import tqdm
from utils.settings import settings
import pickle
from icecream import ic
from torch_geometric.data.hetero_data import Data
from torch.utils.data import Dataset

class ESNLIDataset(Dataset):
    """
    A dataset for ESNLI
    """

    def __init__(self, path: str, split: str = 'train', frac=1.0, chunk=0):
        assert split in {'train', 'val', 'test'}, 'split must be one of train, val, test'
        assert 0.0 <= frac <= 1.0, 'frac must be between 0 and 1'

        super().__init__()
        self.name = f'esnli_{split}_{chunk}'

        # Load pickle file
        esnli_path = osp.join(path, f'esnli_{frac}')
        suffix = '_1' if split == 'train' else ''
        csv_path = osp.join(path, 'esnli', f'esnli_{split}{suffix}.csv')
        keys = ['Sentences', 'Sentences_embedding', 'Explanation_1', 'Explanation_2', 'Explanation_3', 'gold_label', 'pyg_data']
        self.esnli = dict()

        for k in keys:
            try:
                key_path = osp.join(esnli_path, f'{split}_{k}', f'chunk{chunk}.pkl')
                self.esnli[k] = pickle.load(open(key_path, 'rb'))
                if isinstance(self.esnli[k], torch.Tensor):
                    self.esnli[k] = self.esnli[k]
            except Exception as e:
                pass

    def __len__(self):
        return len(self.esnli['gold_label'])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.esnli.items()}
