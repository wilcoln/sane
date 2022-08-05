import os.path as osp
import pickle

import torch.utils.data
from icecream import ic
from torch.utils.data import Dataset
from src.utils.embeddings import tokenize


class ESNLIDataset(Dataset):
    """
    A dataset for ESNLI
    """

    def __init__(self, path: str, split: str = 'train', frac=1.0, chunk=0):
        assert split in {'train', 'val', 'test'}, 'split must be one of train, val, test'
        assert 0.0 <= frac <= 1.0, 'frac must be between 0 and 1'

        # TODO: remove this hack
        if frac == .25:
            chunk = 14 if chunk == 6 else chunk

        super().__init__()
        self.name = f'ESNLI_{frac}'

        # Load pickle file
        esnli_path = osp.join(path, f'esnli_{frac}')
        keys = ['Sentences', 'Sentences_embedding', 'Explanation_1', 'Explanation_2', 'Explanation_3', 'gold_label',
                'concept_ids']
        string_keys = ['Sentences', 'Explanation_1', 'Explanation_2', 'Explanation_3']

        self.esnli = {}
        for k in keys:
            key_path = osp.join(esnli_path, f'{split}_{k}', f'chunk{chunk}.pkl')
            try:
                self.esnli[k] = pickle.load(open(key_path, 'rb'))
            except Exception as e:
                ic(e, key_path)
            if isinstance(self.esnli[k], torch.Tensor):
                self.esnli[k] = self.esnli[k]

        for k in self.esnli:
            if k in string_keys:
                self.esnli[k] = [str(elt) for elt in self.esnli[k]]

        # Tokenize sentences
        self.esnli['encoded_sentences'] = tokenize(self.esnli['Sentences'])

        # Tokenize explanations
        self.esnli['encoded_explanations'] = tokenize(self.esnli['Explanation_1'])

    def __len__(self):
        return len(self.esnli['gold_label'])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.esnli.items()}
