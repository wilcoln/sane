import torch
from icecream import ic
from torch import nn
from torch_geometric.utils import subgraph

from src.conceptnet import conceptnet
from src.settings import settings
from src.utils.nn import GNN, singles_to_triples


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gnn = GNN(hidden_channels=settings.hidden_dim)

    def forward(self, inputs):
        # trainable gnn encoder
        subset = torch.unique(torch.cat(inputs['concept_ids'], dim=0))
        x = conceptnet.concept_embedding[subset]
        edge_index, edge_attr = subgraph(subset, conceptnet.edge_index, conceptnet.edge_attr, relabel_nodes=True)[0]
        x, edge_index, edge_attr = self.gnn(x, edge_index)
        inputs['Knowledge_embedding'] = singles_to_triples(x, edge_index, edge_attr)
        return inputs


if __name__ == '__main__':
    encoder = Encoder()
    inputs = {'Sentences_embedding': torch.randn(10, settings.sent_dim),
              'Knowledge_embedding': torch.randn(10, settings.hidden_dim),
              'concept_ids': [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]}
    out = encoder(inputs)
    ic(out)
