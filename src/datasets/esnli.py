import os.path as osp
import pickle

import math
import torch.utils.data
from icecream import ic
from torch.utils.data import Dataset, default_collate, DataLoader, ConcatDataset
from src.utils.embeddings import tokenize
from src.settings import settings


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
        keys = ['Sentences', 'Sentences_embedding', 'concept_ids', 'gold_label',
                'Explanation_1']  # 'Explanation_2', 'Explanation_3', 'gold_label',
        string_keys = ['Sentences', 'Explanation_1']

        self.esnli = {}
        for k in keys:
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


# DataLoader Helper functions
# Load dataset splits
og_sizes = {'train': 549367, 'val': 9842, 'test': 9824}
new_sizes = {split: int(og_size * settings.data_frac) for split, og_size in og_sizes.items()}
num_chunks = {split: math.ceil(new_size / settings.chunk_size) for split, new_size in new_sizes.items()}

def collate_fn(batch):
    elem = batch[0]

    def collate(key):
        if key in {'Sentences', 'Explanation_1'}:
            return tokenize([d[key] for d in batch])
        if key in {'concept_ids'}:
            return [torch.LongTensor(d[key]) for d in batch]
        return default_collate([d[key] for d in batch])

    return {key: collate(key) for key in elem}

def get_loader(split):
    datasets = [
        ESNLIDataset(path=settings.data_dir, split=split, frac=settings.data_frac, chunk=chunk)
        for chunk in range(num_chunks[split])
    ]

    return DataLoader(ConcatDataset(datasets),
                      batch_size=settings.batch_size, shuffle=False,
                      num_workers=settings.num_workers,
                      collate_fn=collate_fn)

