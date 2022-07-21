import pandas as pd
import os
import os.path as osp

import torch
import pickle
from icecream import ic
from torch_geometric.data import HeteroData
from tqdm import tqdm
from utils.embeddings import bart
from utils.settings import settings
from utils.types import ChunkedList

conceptnet_path = osp.join(settings.data_dir, 'conceptnet', 'conceptnet-assertions-5.6.0_cleaned.csv')
conceptnet_df = pd.read_csv(conceptnet_path)
for column in {'source', 'target', 'relation'}:
    conceptnet_df[column] = conceptnet_df[column].astype('str')

def get_nodes_and_relations(conceptnet_df):
    source_df = conceptnet_df[['source']].rename(columns={'source': 'name'}).drop_duplicates()
    target_df = conceptnet_df[['target']].rename(columns={'target': 'name'}).drop_duplicates()
    concept_df = pd.concat([source_df, target_df], ignore_index=True).drop_duplicates().reset_index()
    relation_df = conceptnet_df[['relation']].rename(columns={'relation': 'name'}).drop_duplicates().reset_index()
    return concept_df, relation_df


concept_df, relation_df = get_nodes_and_relations(conceptnet_df)

concept_list, relation_list = concept_df['name'].tolist(), relation_df['name'].tolist()

conceptnet_dir = osp.join(settings.data_dir, f'conceptnet')

if not os.path.exists(conceptnet_dir):
    os.mkdir(conceptnet_dir)
concept_embedding_path = osp.join(conceptnet_dir, 'concept_embedding')
relation_embedding_path = osp.join(conceptnet_dir, 'relation_embedding')

try:
    # raise Exception
    ic('Loading encoded concepts')
    concept_embedding = ChunkedList(n=len(concept_list), dirpath=concept_embedding_path)
    relation_embedding = ChunkedList(n=len(concept_list), dirpath=relation_embedding_path)

    concept_embedding = torch.cat(concept_embedding.get_chunks(), dim=0)
    relation_embedding = torch.cat(relation_embedding.get_chunks(), dim=0)
except:
    pass
    # chunk_size = 10000
    # # Encoding concepts
    # ic('Encoding concepts')
    # concept_embedding = ChunkedList(lst=concept_list, num_chunks=len(concept_list)//chunk_size).apply(lambda l : bart(l), dirpath=concept_embedding_path)
    # # concept_embedding = dict(zip(concept_df['name'].tolist(), concept_embedding)) # too big
    # # Encoding relations
    # ic('Encoding relations')
    # relation_embedding = ChunkedList(lst=relation_list, num_chunks=len(relation_list)//chunk_size).apply(lambda l : bart(l), dirpath=relation_embedding_path)

def concept_ids_to_pyg_data(node_ids_list):
    """
    Convert a list of list of conceptnet nodes to pytorch geometric data.
    """
    data_list = []
    for node_ids in tqdm(node_ids_list):
        data = HeteroData()
        node_ids = concept_df[concept_df.index.isin(node_ids)].index
        triples_df = conceptnet_df[conceptnet_df['source'].isin(concept_df['name'].loc[node_ids]) | conceptnet_df['target'].isin(concept_df['name'].loc[node_ids])]

        # Load nodes
        concepts, relations = get_nodes_and_relations(triples_df)
        mapping = dict(zip(concepts['name'], concepts.index))
        data['concept'].num_nodes = len(mapping)
        data['concept'].x = torch.LongTensor([concept_list.index(concept) for concept in concepts['name']])

        # Load edges
        for relation in relations['name']:
            src = [mapping[i] for i in triples_df[triples_df['relation'] == relation]['source'].tolist()]
            dst = [mapping[i] for i in triples_df[triples_df['relation'] == relation]['target'].tolist()]
            weight = torch.FloatTensor(triples_df[triples_df['relation'] == relation]['weight'].tolist())
            rel = torch.LongTensor([relation_list.index(relation)]*len(src))
            data['concept', relation, 'concept'].edge_index = torch.LongTensor([src, dst])
            data['concept', relation, 'concept'].edge_label = torch.hstack((rel, weight))
            # Ignoring weight for now

        data_list.append(data)
    return data_list