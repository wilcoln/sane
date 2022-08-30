from dataclasses import dataclass

import torch
from torch import nn

from src.conceptnet import conceptnet
from src.settings import settings
from src.utils.nn import GNN, singles_to_triples


@dataclass
class EncoderOutput:
    output: torch.Tensor  # encoded knowledge (knowledge_size, sent_dim)
    id: torch.Tensor = None  # id of the knowledge (knowledge_size, 3)
    mask: torch.Tensor = None  # mask of the knowledge (batch_size, knowledge_size)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = GNN(hidden_channels=settings.sent_dim)
        self.lin = nn.Linear(3 * settings.sent_dim, settings.sent_dim)

    def forward(self, inputs):
        # Get subgraph
        nodes, edge_index, edge_attr = inputs['concept_ids'], inputs['edge_index'], inputs['edge_attr']
        edge_relation, edge_weight = edge_attr[:, 0].long(), edge_attr[:, 1]

        # Get initial embeddings from conceptnet
        x = conceptnet.concept_embedding[nodes]
        edge_attr = edge_weight.view(-1, 1) * conceptnet.relation_embedding[edge_relation]
        # edge_attr = conceptnet.relation_embedding[edge_relation]  # ignore precomputed edge_weight

        # Send tensors to device
        x, edge_index, edge_attr = x.to(settings.device), edge_index.to(settings.device), edge_attr.to(settings.device)

        if not settings.no_gnn:
            # Apply GNN
            x = self.gnn(x, edge_index, edge_attr)

        # Add self-loops
        self_loop_index = torch.arange(len(nodes)).view(-1, 1).repeat(1, 2).T.to(settings.device)
        edge_index = torch.hstack((edge_index, self_loop_index))
        self_loop_attr = conceptnet.self_loop_embedding.repeat(len(nodes), 1)
        self_loops = torch.Tensor([conceptnet.self_loop_id]).repeat(len(nodes))
        edge_relation = torch.cat([edge_relation, self_loops])
        edge_attr = torch.vstack((edge_attr, self_loop_attr))

        # Convert to triples and project to sentence dimension
        encoded_triples = self.lin(singles_to_triples(x, edge_index, edge_attr))

        # Get triple ids
        x = nodes.view(-1, 1)
        edge_attr = edge_relation.view(-1, 1)
        triple_ids = singles_to_triples(x, edge_index, edge_attr)

        # Get head and tail ids
        head_ids = triple_ids[:, 0]
        tail_ids = triple_ids[:, 2]

        # Compute sentences -> triples mask
        # mask[i][j] = 1 if head_ids[j] in inputs['sent_concept_ids'][i] or tail_ids[j][2] in inputs[
        # 'sent_concept_ids'][i] else 0
        mask = torch.zeros((len(inputs['sent_concept_ids']), len(triple_ids)), dtype=torch.bool)
        for i, sent_concept_ids in enumerate(inputs['sent_concept_ids']):
            mask[i] = torch.any(head_ids == sent_concept_ids.view(-1, 1), dim=0) | torch.any(
                tail_ids == sent_concept_ids.view(-1, 1), dim=0)
        mask = mask.to(settings.device)

        # mask[i] = torch.logical_or(
        #     torch.isin(head_ids, sent_concept_ids),
        #     torch.isin(tail_ids, sent_concept_ids)
        # )

        if self.training:
            return EncoderOutput(output=encoded_triples, mask=mask)
        else:
            return EncoderOutput(output=encoded_triples, mask=mask, id=triple_ids)
