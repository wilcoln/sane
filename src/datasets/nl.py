import math
import os.path as osp
import pickle

import torch.utils.data
from icecream import ic
from torch.utils.data import Dataset, default_collate, DataLoader, ConcatDataset
from torch_geometric.utils import subgraph

from src.conceptnet import conceptnet
from src.preprocessing.common import compute_concept_ids
from src.settings import settings
from src.utils.embeddings import tokenize, bart
from src.utils.semantic_search import semantic_search

string_keys = {'Sentences', 'Explanation_1', 'Explanation_2', 'Explanation_3'}

class NLDataset(Dataset):
    """
    A Natural Language dataset
    """

    def __init__(self, path: str = None, name: str = 'esnli', split: str = 'train', frac=1.0, chunk=0,
                 data_dict=None):
        assert name in {'esnli', 'comve', 'cose'}, 'Dataset name not supported'
        if data_dict:
            self.nl = data_dict
            # encode sentences
            self.nl['Sentences_embedding'] = bart(self.nl['Sentences']).detach().cpu()
            if not settings.no_knowledge:
                self.nl['concept_ids'] = compute_concept_ids(self.nl['Sentences'])
        else:
            assert path is not None
            assert split in {'train', 'val', 'test'}, 'split must be one of train, val, test'
            assert 0.0 <= frac <= 1.0, 'frac must be between 0 and 1'

            super().__init__()

            # Load pickle file
            self.path = osp.join(path, f'{name}_{frac}')
            keys = ['Sentences', 'Sentences_embedding', 'gold_label', 'Explanation_1']

            if not settings.no_knowledge:
                keys += ['concept_ids']

            if split in {'val', 'test'}:
                keys += ['Explanation_2', 'Explanation_3']

            self.nl = {}
            for k in keys:
                key_path = osp.join(self.path, f'{split}_{k}', f'chunk{chunk}.pkl')
                try:
                    self.nl[k] = pickle.load(open(key_path, 'rb'))
                    if isinstance(self.nl[k], torch.Tensor):
                        self.nl[k] = self.nl[k].detach().cpu()
                except Exception as e:
                    ic(e, key_path)

        for k in self.nl:
            if k in string_keys:
                self.nl[k] = [str(elt) for elt in self.nl[k]]

    def __len__(self):
        return len(self.nl['gold_label'])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.nl.items()}


# DataLoader Helper functions
# Load dataset splits

def collate_fn(batch):
    elem = batch[0]
    concept_ids_key = 'concept_ids'

    # Preliminaries
    inputs = {key: default_collate([d[key] for d in batch]) for key in elem if key != concept_ids_key}

    # Tokenize strings
    for key in (set(elem) & string_keys):
        inputs[key] = tokenize([d[key] for d in batch])

    if not settings.no_knowledge:
        # Get exact concept ids
        exact_sent_concept_ids = [torch.LongTensor(d[concept_ids_key][0]) for d in batch]
        exact_concept_ids = torch.unique(torch.cat(exact_sent_concept_ids, dim=0))

        # Get neighboring concept ids
        neighboring_sent_concept_ids = [torch.LongTensor(d[concept_ids_key][1]) for d in batch]
        neighboring_concept_ids = torch.unique(torch.cat(neighboring_sent_concept_ids, dim=0))

        # Filter neighboring concepts using semantic search
        top_neighboring_concepts_indices = semantic_search(
            queries=inputs['Sentences_embedding'],
            values=conceptnet.concept_embedding[neighboring_concept_ids],
            top_k=settings.max_concepts_per_sent,
        )
        neighboring_concept_ids = neighboring_concept_ids[top_neighboring_concepts_indices]

        # Merge exact and neighboring concept ids at sentence-level
        sent_concept_ids = [
            torch.cat([exact_sent_concept_ids[i], neighboring_concept_ids[i]])
            for i in range(len(batch))
        ]
        # exact_sent_concept_ids = torch.cat(exact_sent_concept_ids, dim=1)
        # neighboring_sent_concept_ids = torch.cat(neighboring_sent_concept_ids, dim=1)
        # ic(exact_sent_concept_ids.shape, neighboring_sent_concept_ids.shape)
        # sent_concept_ids = torch.hstack((exact_sent_concept_ids, neighboring_sent_concept_ids))

        # Merge exact and neighboring concept ids and filter out duplicates at batch-level
        concept_ids = torch.cat([exact_concept_ids.flatten(), neighboring_concept_ids.flatten()], dim=0)
        concept_ids = torch.unique(concept_ids)

        # Finalize batch
        inputs[f'sent_{concept_ids_key}'] = sent_concept_ids
        inputs[concept_ids_key] = concept_ids
        inputs['edge_index'], inputs['edge_attr'] = subgraph(
            subset=concept_ids,
            edge_index=conceptnet.pyg.edge_index,
            edge_attr=conceptnet.pyg.edge_attr,
            relabel_nodes=True,
        )

    return inputs


def get_dataset(split: str, name: str):
    if name == 'esnli':
        og_sizes = {'train': 549367, 'val': 9842, 'test': 9824}
    elif name == 'comve':
        og_sizes = {'train': 10000, 'val': 1000, 'test': 1000}
    elif name == 'cose':
        og_sizes = {'train': 9741, 'val': 1221}
    else:
        raise ValueError(f'{name} is not a valid dataset name')

    new_sizes = {split: int(og_size * settings.data_frac) for split, og_size in og_sizes.items()}
    num_chunks = {split: math.ceil(new_size / settings.chunk_size) for split, new_size in new_sizes.items()}

    datasets = [
        NLDataset(path=settings.data_dir, name=name, split=split, frac=settings.data_frac, chunk=chunk)
        for chunk in range(num_chunks[split])
    ]
    return ConcatDataset(datasets)


def get_loader(split):
    dataset = get_dataset(split, name=settings.dataset)
    return DataLoader(dataset,
                      batch_size=settings.batch_size, shuffle=False,
                      num_workers=settings.num_workers,
                      collate_fn=collate_fn)


def get_sanity_check_loader():
    dataset = NLDataset(path=settings.data_dir, split='train')
    dataset = torch.utils.data.Subset(dataset, list(range(10)))
    return DataLoader(dataset,
                      batch_size=settings.batch_size, shuffle=False,
                      num_workers=settings.num_workers,
                      collate_fn=collate_fn)
