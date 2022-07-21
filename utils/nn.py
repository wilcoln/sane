import torch
from icecream import ic
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero, global_add_pool, HeteroConv, Linear
from utils.graphs import concept_embedding



class MLP(nn.Module):
    """ Multi-layer perceptron. """

    def __init__(self, in_channels, hidden_channels=100, out_channels=None, num_layers=2, batch_norm=False):
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
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers-1)] +
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

        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, data):
        x, edge_index = concept_embedding[data.x], data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        # x = global_add_pool(x, data.batch)
        # x = self.mlp(x)
        return x, edge_index


class HeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels=None, num_layers=2):
        super().__init__()
        out_channels = out_channels if out_channels is not None else hidden_channels

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            # return self.lin(x_dict['author'])
        # x = global_add_pool(x['concept'], data.batch)
        return x_dict, edge_index_dict


def singles_to_triples(x, edge_index):
    heads = x[edge_index[0]]  # (E, D)
    tails = x[edge_index[1]]  # (E, D)

    # Add triples
    triples = torch.cat([heads, tails], dim=1)  # (E, 2D)
    # # add inverse triples
    # inverse_triples = torch.cat([tails, heads], dim=1)  # (E, 2*D)
    # triples = torch.cat([triples, inverse_triples], dim=0)  # (2E, 2D)

    # # add self-loops
    # self_loops = torch.cat([data, data], dim=1)  # (V, D)
    # triples = torch.cat([triples, self_loops], dim=0)  # (2E + V, 2D)

    return triples
