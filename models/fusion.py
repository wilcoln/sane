import pandas as pd
from torch import nn
from icecream import ic
import torch
import os.path as osp
from utils.embeddings import bart
from utils.settings import settings
from utils.nn import HeteroGNN, GNN, singles_to_triples


class Fuser(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lin = nn.Linear(2*384 + 2*settings.hidden_dim, 1)

    def forward(self, inputs):
        for i, encoded_triples in enumerate(inputs['Knowledge_embedding']):
            sentence_queries = torch.cat([inputs['Sentences_embedding'][i].repeat(encoded_triples.shape[0], 1),
                                  encoded_triples], dim=1)
            
            scores = self.lin(sentence_queries)
            attn_scores = torch.softmax(scores, dim=1)
            inputs['Knowledge_embedding'][i] = torch.sum(attn_scores * encoded_triples, dim=0).unsqueeze(0)

        inputs['Knowledge_embedding'] = torch.cat(inputs['Knowledge_embedding'], dim=0)

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
        inputs['Knowledge_embedding'] = [singles_to_triples(*self.gnn(data.to_homogeneous())) for
                                         data in inputs['pyg_data']]
        return inputs
