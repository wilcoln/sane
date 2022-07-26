from utils.graphs import conceptnet_df, concept_df, relation_df, concept_dict, relation_dict
import os.path as osp
import torch
from icecream import ic
from torch_geometric.data import HeteroData
from utils.settings import settings

data = HeteroData()
node_ids = concept_df.index
triples_df = conceptnet_df[conceptnet_df['source'].isin(concept_df['name'].loc[node_ids]) | conceptnet_df['target'].isin(concept_df['name'].loc[node_ids])]
# Load nodes
data['concept'].num_nodes = len(concept_dict)
data['concept'].x = torch.LongTensor([concept_dict[concept] for concept in concept_df['name']])
# Load edges
for relation in relation_df['name']:
    src = [concept_dict[i] for i in triples_df[triples_df['relation'] == relation]['source'].tolist()]
    dst = [concept_dict[i] for i in triples_df[triples_df['relation'] == relation]['target'].tolist()]
    weight = torch.FloatTensor(triples_df[triples_df['relation'] == relation]['weight'].tolist())
    rel = torch.LongTensor([relation_dict[relation]]*len(src))
    data['concept', relation, 'concept'].edge_index = torch.LongTensor([src, dst])
    data['concept', relation, 'concept'].edge_label = torch.hstack((rel, weight))
    # Ignoring weight for now


conceptnet_pyg_dir = osp.join(settings.data_dir, 'conceptnet', 'conceptnet.pyg')

torch.save(data, conceptnet_pyg_dir)
