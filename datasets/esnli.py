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

    def __init__(self, path: str, split: str = 'train', frac=1.0):
        assert split in {'train', 'val', 'test'}, 'split must be one of train, val, test'
        assert 0.0 <= frac <= 1.0, 'frac must be between 0 and 1'

        super().__init__()
        self.name = f'esnli_{split}'

        # Load pickle file
        esnli_path = osp.join(path, f'{self.name}.pkl')
        self.esnli = pickle.load(open(esnli_path, 'rb'))

    def __len__(self):
        return len(self.esnli['gold_label'])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.esnli.items() if k not in {'Triple_ids'}}
