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
        self.k_proj = nn.Linear(2*settings.hidden_dim, settings.hidden_dim)
        self.s_proj = nn.Linear(settings.sent_dim, settings.hidden_dim)
    def forward(self, inputs):
        # Project sentences
        S = self.s_proj(inputs['Sentences_embedding'])
        for i, encoded_triples in enumerate(inputs['Knowledge_embedding']):
            # Project Knowledge for Sentence i
            K = self.k_proj(encoded_triples)
            scores = K @ S[i].unsqueeze(1) 
            attn_scores = torch.softmax(scores, dim=1)
            inputs['Knowledge_embedding'][i] = torch.sum(attn_scores * encoded_triples, dim=0).unsqueeze(0)

        inputs['Knowledge_embedding'] = torch.cat(inputs['Knowledge_embedding'], dim=0)

        return inputs

# class Fuser(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.align = nn.Linear(3*settings.hidden_dim, 1)
#         self.s_proj = nn.Linear(settings.sent_dim, settings.hidden_dim)
#     def forward(self, inputs):
#         S = self.s_proj(inputs['Sentences_embedding'])
#         for i, encoded_triples in enumerate(inputs['Knowledge_embedding']):
#             sentence_knowledge = torch.cat([
#                 S[i].repeat(encoded_triples.shape[0], 1),
#                 encoded_triples
#             ], dim=1)
            
#             align_scores = self.align(sentence_knowledge)
#             attn_scores = torch.softmax(align_scores, dim=1)
#             inputs['Knowledge_embedding'][i] = torch.sum(attn_scores * encoded_triples, dim=0).unsqueeze(0)

#         inputs['Knowledge_embedding'] = torch.cat(inputs['Knowledge_embedding'], dim=0)

#         return inputs


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gnn = GNN(hidden_channels=settings.hidden_dim)  # HeteroGNN(metadata=kwargs['metadata'],
        # hidden_channels=32)

    def forward(self, inputs):
        # trainable gnn encoder
        inputs['Knowledge_embedding'] = [singles_to_triples(*self.gnn(data)) for
                                         data in inputs['pyg_data']]
        return inputs
