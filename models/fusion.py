import pandas as pd
from torch import nn

import os.path as osp
from utils.embeddings import bart
from utils.settings import settings
from utils.nn import GNN
from utils.graphs import triple_ids_to_pyg_data

class Fuser(nn.Module):
    def forward(self, inputs):
        return inputs


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gnn = GNN(hidden_channels=32)

    def forward(self, inputs):
        # frozen lm encoder
        if 'Sentences_embeddings' not in inputs:
            inputs['Sentences'] = [f'{sent1} -> {sent2}' for sent1, sent2 in zip(inputs['Sentence1'], inputs['Sentence2'])]
            inputs['Sentences_embeddings'] = bart(inputs['Sentences'])
        # trainable gnn encoder
        inputs['pyg_data'] = triple_ids_to_pyg_data(inputs['Triple_ids'])
        inputs['Knowledge_embedding'] = self.gnn(inputs['pyg_data'])


        ic(inputs['Knowledge_mebedding'].shape)
        ic(inputs['Knowledge_embedding'])
        return inputs
