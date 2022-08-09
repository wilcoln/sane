import math
import os.path as osp
import pickle

import torch.utils.data
from icecream import ic
from torch.utils.data import Dataset, default_collate, DataLoader, ConcatDataset
from torch_geometric.utils import subgraph

from src.conceptnet import conceptnet
from src.preprocessing.esnli import compute_concept_ids
from src.settings import settings
from src.utils.embeddings import tokenize, bart
from src.utils.semantic_search import semantic_search

string_keys = {'Sentences', 'Explanation_1', 'Explanation_2', 'Explanation_3'}


class ESNLIDataset(Dataset):
    """
    A dataset for ESNLI
    """

    def __init__(self, path: str = None, split: str = 'train', frac=1.0, chunk=0, data_dict=None):
        if data_dict:
            self.esnli = data_dict
            # encode sentences
            self.esnli['Sentences_embedding'] = bart(self.esnli['Sentences'])
            self.esnli['concept_ids'] = compute_concept_ids(self.esnli['Sentences'])
        else:
            assert path is not None
            assert split in {'train', 'val', 'test'}, 'split must be one of train, val, test'
            assert 0.0 <= frac <= 1.0, 'frac must be between 0 and 1'

            super().__init__()
            self.name = f'ESNLI_{frac}'

            # Load pickle file
            esnli_path = osp.join(path, f'esnli_{frac}')
            keys = ['Sentences', 'Sentences_embedding', 'concept_ids', 'gold_label', 'Explanation_1']
            if split in {'val', 'test'}:
                keys += ['Explanation_2', 'Explanation_3']

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

    def collate_key(key):
        if key == 'concept_ids':
            return [torch.LongTensor(d[key]) for d in batch]
        return default_collate([d[key] for d in batch])

    # Preliminaries
    inputs = {key: collate_key(key) for key in elem}

    # Tokenize strings
    for key in (set(elem) & string_keys):
        inputs[key] = tokenize([d[key] for d in batch])

    # Curate sub-knowledge using semantic search
    concept_ids = torch.unique(torch.cat(inputs['concept_ids'], dim=0))
    top_concepts_indices = semantic_search(
        queries=inputs['Sentences_embedding'],
        values=conceptnet.concept_embedding[concept_ids],
        top_k=settings.max_concepts_per_sent,
    )

    concept_ids = concept_ids[top_concepts_indices]
    concept_ids = torch.unique(concept_ids.squeeze())

    # Finalize batch
    inputs['concept_ids'], inputs['edge_index'], inputs['edge_attr'] = concept_ids, *subgraph(
        subset=concept_ids,
        edge_index=conceptnet.pyg.edge_index,
        edge_attr=conceptnet.pyg.edge_attr,
        relabel_nodes=True
    )

    return inputs


def get_dataset(split: str):
    datasets = [
        ESNLIDataset(path=settings.data_dir, split=split, frac=settings.data_frac, chunk=chunk)
        for chunk in range(num_chunks[split])
    ]
    return ConcatDataset(datasets)


def get_loader(split, dataset=None):
    dataset = get_dataset(split) if dataset is None else dataset
    return DataLoader(dataset,
                      batch_size=settings.batch_size, shuffle=False,
                      num_workers=settings.num_workers,
                      collate_fn=collate_fn)


def get_sanity_check_loader():
    dataset = ESNLIDataset(path=settings.data_dir, split='train')
    dataset = torch.utils.data.Subset(dataset, list(range(10)))
    return DataLoader(dataset,
                      batch_size=settings.batch_size, shuffle=False,
                      num_workers=settings.num_workers,
                      collate_fn=collate_fn)
