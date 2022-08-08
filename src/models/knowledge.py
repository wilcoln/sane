from dataclasses import dataclass

import torch
from torch import nn

from src.conceptnet import conceptnet
from src.settings import settings
from src.utils.nn import GNN, singles_to_triples


@dataclass
class EncoderOutput:
    embed: torch.Tensor  # encoded knowledge (knowledge_size, sent_dim)
    id: torch.Tensor = None  # id of the knowledge (knowledge_size, 3)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = GNN(hidden_channels=settings.sent_dim // 3)

    def forward(self, inputs):
        # trainable gnn encoder
        nodes, edge_index, edge_attr = inputs['concept_ids'], inputs['edge_index'], inputs['edge_attr']
        edge_relation, edge_weight = edge_attr[:, 0].long(), edge_attr[:, 1]

        # Encode triples
        x = conceptnet.concept_embedding[nodes]
        edge_attr = edge_weight.view(-1, 1) * conceptnet.relation_embedding[edge_relation]
        x, edge_index, edge_attr = self.gnn(x, edge_index, edge_attr)
        encoded_triples = singles_to_triples(x, edge_index, edge_attr)

        if self.training:
            return EncoderOutput(embed=encoded_triples)
        else:
            # Return triple ids too
            x = nodes.view(-1, 1)
            edge_attr = edge_relation.view(-1, 1)
            triple_ids = singles_to_triples(x, edge_index, edge_attr)
            return EncoderOutput(embed=encoded_triples, id=triple_ids)
