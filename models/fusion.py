import pandas as pd
from torch import nn
from icecream import ic
import torch
import os.path as osp
from utils.embeddings import bart
from utils.settings import settings
from utils.nn import HeteroGNN, GNN

class Fuser(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs):
        return inputs


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gnn = GNN(hidden_channels=settings.hidden_dim)  # HeteroGNN(metadata=kwargs['metadata'],
        # hidden_channels=32)

    def forward(self, inputs):
        # frozen lm encoder
        if 'Sentences_embedding' not in inputs:
            inputs['Sentences_embedding'] = torch.cat([inputs['Sentence1_embedding'], inputs['Sentence2_embedding']], dim=1)
        # trainable gnn encoder
        inputs['Knowledge_embedding'] = torch.cat([self.gnn(data.to_homogeneous()) for data in inputs['pyg_data']], dim=0)
        return inputs
