import pandas as pd
from torch import nn
from icecream import ic
import torch
import os.path as osp
from utils.embeddings import bart
from utils.settings import settings
from utils.nn import HeteroGNN, GNN, singles_to_triples
from datasets.esnli import conceptnet, concept_embedding
from torch_geometric.utils import subgraph


# class Fuser(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.k_proj = nn.Linear(2*settings.hidden_dim, settings.hidden_dim)
#         self.s_proj = nn.Linear(settings.sent_dim, settings.hidden_dim)
#     def forward(self, inputs):
#         # Project sentences
#         S = self.s_proj(inputs['Sentences_embedding'])
#         attns = []
#         for i, encoded_triples in enumerate(inputs['Knowledge_embedding']):
#             # Project Knowledge for Sentence i
#             K = self.k_proj(encoded_triples)
#             scores = K @ S[i].unsqueeze(1) 
#             attn_scores = torch.softmax(scores, dim=1)
#             inputs['Knowledge_embedding'][i] = torch.sum(attn_scores * encoded_triples, dim=0).unsqueeze(0)
#             attns.append(attn_scores)

#         inputs['Knowledge_embedding'] = torch.cat(inputs['Knowledge_embedding'], dim=0)

#         return attns, inputs


class Fuser(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.k_proj = nn.Linear(2*settings.hidden_dim, settings.hidden_dim)
        self.s_proj = nn.Linear(settings.sent_dim, settings.hidden_dim)
    def forward(self, inputs):
        # Project sentences
        S = self.s_proj(inputs['Sentences_embedding'])
        K = self.k_proj(inputs['Knowledge_embedding'])
        V = inputs['Knowledge_embedding']
        scores = S @ K.T
        attns = torch.softmax(scores, dim=0)
        inputs['Knowledge_embedding'] = attns @ V
        return attns, inputs


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gnn = GNN(hidden_channels=settings.hidden_dim)  # HeteroGNN(metadata=kwargs['metadata'],
        self.conceptnet = conceptnet.to_homogeneous()
        # hidden_channels=32)

    def forward(self, inputs):
        # trainable gnn encoder
        node_ids_list = [e.x[torch.randperm(e.x.size(0))[:200]] for e in inputs['pyg_data']]
        subset = torch.unique(torch.cat(node_ids_list, dim=0))
        x = concept_embedding[subset]
        edge_index = subgraph(subset, self.conceptnet.edge_index, relabel_nodes=True)[0]
        x, edge_index = self.gnn(x, edge_index)
        inputs['Knowledge_embedding'] = singles_to_triples(x, edge_index)
        return inputs


class HeteroEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gnn = HeteroGNN(metadata=kwargs['metadata'], hidden_channels=settings.hidden_dim)

    def forward(self, inputs):
        # trainable gnn encoder
        node_ids_list = [e['concept'].x[torch.randperm(e['concept'].x.size(0))[:200]] for e in inputs['pyg_data']]
        subset = torch.unique(torch.cat(node_ids_list, dim=0))
        subset_dict = {'concept': subset}
        data = subgraph(conceptnet, subset_dict)
        data['concept'].x = concept_embedding[subset]
        x_dict, edge_index_dict = self.gnn(data)
        inputs['Knowledge_embedding'] = singles_to_triples(x_dict, edge_index_dict)
        return inputs