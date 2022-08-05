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
        concept_ids_list = [
            concept_ids[torch.randperm(concept_ids.shape[0])[:settings.max_concepts_per_sent]]
            for concept_ids in inputs['concept_ids']
        ]
        subset = torch.unique(torch.cat(concept_ids_list, dim=0))
        edge_index, edge_attr = subgraph(subset, conceptnet.pyg.edge_index, conceptnet.pyg.edge_attr,
                                         relabel_nodes=True)
        edge_relation, edge_weight = edge_attr[:, 0].long(), edge_attr[:, 1]

        x = conceptnet.concept_embedding[subset]
        edge_attr = edge_weight.view(-1, 1) * conceptnet.relation_embedding[edge_relation]
        x, edge_index, edge_attr = self.gnn(x, edge_index, edge_attr)
        inputs['Knowledge_embedding'] = singles_to_triples(x, edge_index, edge_attr)
        return inputs


if __name__ == '__main__':
    encoder = Encoder()
    inputs = {'Sentences_embedding': torch.randn(2, settings.sent_dim),
              'concept_ids': [torch.tensor([0, 1, 2, 3, 10, 11, 1005]),
                              torch.tensor([0, 1, 3, 4, 5, 6, 1001, 1002, 1003])]}
    out = encoder(inputs)
    ic(out)
