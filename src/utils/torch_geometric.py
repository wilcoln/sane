from typing import Union, Tuple, List, Dict, Optional

import torch
from torch import Tensor
from torch_geometric.data.storage import (NodeStorage,
                                          EdgeStorage)
from torch_geometric.typing import NodeType, EdgeType, PairTensor
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes

NodeOrEdgeType = Union[NodeType, EdgeType]
NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]


def bipartite_subgraph(
        subset: Union[PairTensor, Tuple[List[int], List[int]]],
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        relabel_nodes: bool = False,
        size: Tuple[int, int] = None,
        return_edge_mask: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Returns the induced subgraph of the bipartite graph
    :obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.

    Args:
        subset (Tuple[Tensor, Tensor] or tuple([int],[int])): The nodes
            to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        size (tuple, optional): The number of nodes.
            (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    device = edge_index.device

    if isinstance(subset[0], (list, tuple)):
        subset = (torch.tensor(subset[0], dtype=torch.long, device=device),
                  torch.tensor(subset[1], dtype=torch.long, device=device))

    if subset[0].dtype == torch.bool or subset[0].dtype == torch.uint8:
        size = subset[0].size(0), subset[1].size(0)
    else:
        if size is None:
            size = (maybe_num_nodes(edge_index[0]),
                    maybe_num_nodes(edge_index[1]))
        subset = (index_to_mask(subset[0], size=size[0]),
                  index_to_mask(subset[1], size=size[1]))

    node_mask = subset
    edge_mask = node_mask[0][edge_index[0]] & node_mask[1][edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx_i = torch.zeros(node_mask[0].size(0), dtype=torch.long,
                                 device=device)
        node_idx_j = torch.zeros(node_mask[1].size(0), dtype=torch.long,
                                 device=device)
        node_idx_i[node_mask[0]] = torch.arange(node_mask[0].sum().item(),
                                                device=device)
        node_idx_j[node_mask[1]] = torch.arange(node_mask[1].sum().item(),
                                                device=device)
        edge_index = torch.stack(
            [node_idx_i[edge_index[0]], node_idx_j[edge_index[1]]])

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


def hetero_subgraph(heterodata, subset_dict: Dict[NodeType, Tensor]) -> 'HeteroData':
    r"""Returns the induced subgraph containing the node types and
    corresponding nodes in :obj:`subset_dict`.

    .. code-block:: python

        data = HeteroData()
        data['paper'].x = ...
        data['author'].x = ...
        data['conference'].x = ...
        data['paper', 'cites', 'paper'].edge_index = ...
        data['author', 'paper'].edge_index = ...
        data['paper', 'conference'].edge_index = ...
        print(data)
        >>> HeteroData(
            paper={ x=[10, 16] },
            author={ x=[5, 32] },
            conference={ x=[5, 8] },
            (paper, cites, paper)={ edge_index=[2, 50] },
            (author, to, paper)={ edge_index=[2, 30] },
            (paper, to, conference)={ edge_index=[2, 25] }
        )

        subset_dict = {
            'paper': torch.tensor([3, 4, 5, 6]),
            'author': torch.tensor([0, 2]),
        }

        print(subgraph(data, subset_dict))
        >>> HeteroData(
            paper={ x=[4, 16] },
            author={ x=[2, 32] },
            (paper, cites, paper)={ edge_index=[2, 24] },
            (author, to, paper)={ edge_index=[2, 5] }
        )

    Args:
        subset_dict (Dict[str, LongTensor or BoolTensor]): A dictonary
            holding the nodes to keep for each node type.
    """
    data = heterodata.__class__(heterodata._global_store)

    for node_type, subset in subset_dict.items():
        for key, value in heterodata[node_type].items():
            if key == 'num_nodes':
                if subset.dtype == torch.bool:
                    data[node_type].num_nodes = int(subset.sum())
                else:
                    data[node_type].num_nodes = subset.size(0)
            elif heterodata[node_type].is_node_attr(key):
                data[node_type][key] = value[subset]
            else:
                data[node_type][key] = value

    for edge_type in heterodata.edge_types:
        src, _, dst = edge_type
        if src not in subset_dict or dst not in subset_dict:
            continue

        edge_index, _, edge_mask = bipartite_subgraph(
            (subset_dict[src], subset_dict[dst]),
            heterodata[edge_type].edge_index,
            relabel_nodes=True,
            size=(heterodata[src].num_nodes, heterodata[dst].num_nodes),
            return_edge_mask=True,
        )

        for key, value in heterodata[edge_type].items():
            if key == 'edge_index':
                data[edge_type].edge_index = edge_index
            elif heterodata[edge_type].is_edge_attr(key):
                data[edge_type][key] = value[edge_mask]
            else:
                data[edge_type][key] = value

    return data


# if __name__ == '__main__':
#     from src.datasets.esnli import conceptnet
#
#     print(conceptnet)
#
#     subset_dict = {
#         'concept': torch.tensor([3, 4, 5, 6, 102, 456]),
#     }
#
#     print(hetero_subgraph(conceptnet, subset_dict))
