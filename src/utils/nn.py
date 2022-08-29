import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

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


class GNN(nn.Module):
    def __init__(self, hidden_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else hidden_channels

        self.conv1 = GATConv((-1, -1), hidden_channels, edge_dim=settings.sent_dim).to(settings.device)
        self.conv2 = GATConv((-1, -1), out_channels, edge_dim=hidden_channels).to(settings.device)

    def forward(self, x, edge_index, edge_attr):
        x, edge_index, edge_attr = x.to(settings.device), edge_index.to(settings.device), edge_attr.to(settings.device)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x


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


def unfreeze(model):
    # unfreeze parameters
    for param in model.parameters():
        param.requires_grad = True


def freeze_modules(models):
    for model in models:
        freeze(model)


def unfreeze_modules(models):
    for model in models:
        unfreeze(model)
