import torch
from torch import nn
from torch_geometric.utils import subgraph
from src.utils.torch_geometric import hetero_subgraph

from src.conceptnet import conceptnet
from src.utils.nn import HeteroGNN, GNN, singles_to_triples
from src.utils.settings import settings


class Fuser(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.k_proj = nn.Linear(2 * settings.hidden_dim, settings.hidden_dim)
        self.s_proj = nn.Linear(settings.sent_dim, settings.hidden_dim)

    def forward(self, inputs):
        # Project sentences
        S = self.s_proj(inputs['Sentences_embedding'])
        K = self.k_proj(inputs['Knowledge_embedding'])
        V = inputs['Knowledge_embedding']
        align_scores = S @ K.T
        attn_scores = torch.softmax(align_scores, dim=0)
        inputs['Knowledge_embedding'] = attn_scores @ V
        return attn_scores, inputs


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
        x = conceptnet.concept_embedding[subset]
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
        # node_ids_list = [e['concept'].x[torch.randperm(e['concept'].x.size(0))[:200]] for e in inputs['pyg_data']]
        node_ids_list = [e['concept'].x for e in inputs['pyg_data']]
        subset = torch.unique(torch.cat(node_ids_list, dim=0))
        subset_dict = {'concept': subset}
        data = hetero_subgraph(conceptnet, subset_dict)
        data['concept'].x = conceptnet.concept_embedding[subset]
        x_dict, edge_index_dict = self.gnn(data)
        inputs['Knowledge_embedding'] = singles_to_triples(x_dict, edge_index_dict)
        return inputs
