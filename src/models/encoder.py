import torch
from torch_geometric.utils import subgraph
from src.utils.torch_geometric import hetero_subgraph
from torch import nn
from src.settings import settings

from src.conceptnet import conceptnet
from src.utils.nn import HeteroGNN, GNN, singles_to_triples


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gnn = GNN(hidden_channels=settings.hidden_dim)  # HeteroGNN(metadata=kwargs['metadata'],
        self.conceptnet = conceptnet.pyg.to_homogeneous()
        # hidden_channels=32)

    def forward(self, inputs):
        # trainable gnn encoder
        # node_ids_list = [e.x[torch.randperm(e.x.size(0))[:200]] for e in inputs['pyg_data']]
        subset = torch.unique(torch.cat(inputs['concept_ids'], dim=0))
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
        subset = torch.unique(torch.cat(inputs['concept_ids'], dim=0))
        subset_dict = {'concept': subset}
        data = hetero_subgraph(conceptnet, subset_dict)
        data['concept'].x = conceptnet.concept_embedding[subset]
        x_dict, edge_index_dict = self.gnn(data)
        inputs['Knowledge_embedding'] = singles_to_triples(x_dict, edge_index_dict)
        return inputs
