from typing import Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import GeneralConv
from torch_geometric.typing import OptPairTensor, Adj, Size

from src.settings import settings


class MLP(nn.Module):
    """ Multi-layer perceptron. """

    def __init__(self, in_channels, hidden_channels=100, out_channels=None, num_layers=1, batch_norm=False):
        assert num_layers >= 1, "num_layers must be at least 1"
        super(MLP, self).__init__()

        self.name = f'{num_layers}-MLP'
        self.batch_norm = batch_norm

        out_channels = out_channels if out_channels is not None else hidden_channels

        self.layers = nn.ModuleList(
            [nn.Linear(in_channels, hidden_channels)] +
            [
                nn.Linear(hidden_channels, hidden_channels)
                for _ in range(num_layers - 2)
            ] +
            [nn.Linear(hidden_channels, out_channels)]
        )

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)] +
            [nn.BatchNorm1d(out_channels)]
        )

    def forward(self, x):
        for i, (layer, batch_norm) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            if self.batch_norm:
                x = batch_norm(x)
            if i < len(self.layers) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


class ConceptNetConv(GeneralConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: Tensor = None, size: Size = None) -> Tuple[Tensor, Tensor]:
        x_out = super().forward(x, edge_index, edge_attr, size)
        edge_out = self.lin_edge(edge_attr)
        return x_out, edge_out


class GNN(nn.Module):
    def __init__(self, hidden_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else hidden_channels

        self.conv1 = ConceptNetConv((-1, -1), hidden_channels, in_edge_channels=settings.sent_dim).to(settings.device)
        self.conv2 = ConceptNetConv((-1, -1), out_channels, in_edge_channels=hidden_channels).to(settings.device)

    def forward(self, x, edge_index, edge_attr):
        x, edge_index, edge_attr = x.to(settings.device), edge_index.to(settings.device), edge_attr.to(settings.device)
        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = F.relu(x), F.relu(edge_attr)
        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        return x, edge_index, edge_attr


def singles_to_triples(x, edge_index, edge_attr):
    heads = x[edge_index[0]]  # (E, D)
    tails = x[edge_index[1]]  # (E, D)
    relations = edge_attr

    # Return triples
    return torch.cat([heads, relations, tails], dim=1)  # (E, 2D)

def freeze(model):
    # freeze parameters
    for param in model.parameters():
      param.requires_grad = False
    # prepare layers for evaluation
    model.eval()
    return model