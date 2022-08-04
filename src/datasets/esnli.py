import os.path as osp
import pickle

import torch.utils.data
from icecream import ic
from torch.utils.data import Dataset
from transformers import BartTokenizer

from ..utils.settings import settings
from ..utils.types import ChunkedList

conceptnet_dir = osp.join(settings.data_dir, f'conceptnet')
concept_embedding = ChunkedList(n=5779, dirpath=osp.join(conceptnet_dir, 'concept_embedding'))
concept_embedding = torch.cat(concept_embedding.get_chunks(), dim=0)
conceptnet = torch.load(osp.join(conceptnet_dir, 'conceptnet.pyg'))
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


class ESNLIDataset(Dataset):
    """
    A dataset for ESNLI
    """

    def __init__(self, path: str, split: str = 'train', frac=1.0, chunk=0):
        assert split in {'train', 'val', 'test'}, 'split must be one of train, val, test'
        assert 0.0 <= frac <= 1.0, 'frac must be between 0 and 1'
        chunk = 14 if chunk == 6 else chunk

        super().__init__()
        self.name = f'esnli_{split}_{chunk}'

        # Load pickle file
        esnli_path = osp.join(path, f'esnli_{frac}')
        suffix = '_1' if split == 'train' else ''
        csv_path = osp.join(path, 'esnli', f'esnli_{split}{suffix}.csv')
        keys = ['Sentences', 'Sentences_embedding', 'Explanation_1', 'Explanation_2', 'Explanation_3', 'gold_label',
                'pyg_data']
        string_keys = ['Sentences', 'Explanation_1', 'Explanation_2', 'Explanation_3']
        self.esnli = dict()

        for k in keys:
            try:
                key_path = osp.join(esnli_path, f'{split}_{k}', f'chunk{chunk}.pkl')
                self.esnli[k] = pickle.load(open(key_path, 'rb'))
                if isinstance(self.esnli[k], torch.Tensor):
                    self.esnli[k] = self.esnli[k]
            except Exception as e:
                ic(e, key_path)

        for k in self.esnli:
            # # Reduce the dataset
            # self.esnli[k] = self.esnli[k][:len(self.esnli[k])//5]
            if k in string_keys:
                self.esnli[k] = [str(elt) for elt in self.esnli[k]]

        # self.esnli['pyg_data'] = [conceptnet.subgraph({'concept': torch.Tensor(v)}) for v in self.esnli['concept_ids']]
        self.esnli['pyg_data'] = [data.to_homogeneous() for data in self.esnli['pyg_data']]

    def __len__(self):
        return len(self.esnli['gold_label'])

    def __getitem__(self, i):
        try:
            return {k: v[i] for k, v in self.esnli.items()}
        except:
            return self[i - 1]
