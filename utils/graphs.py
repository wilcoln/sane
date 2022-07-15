import pandas as pd
import os.path as osp

import torch
import pickle
from icecream import ic
from torch_geometric.data import HeteroData
from tqdm import tqdm
from utils.embeddings import bart
from utils.settings import settings

conceptnet_path = osp.join(settings.data_dir, 'conceptnet', 'conceptnet-assertions-5.6.0_cleaned.csv')
conceptnet_df = pd.read_csv(conceptnet_path)
conceptnet_df = conceptnet_df.sample(frac=.01, random_state=0)
conceptnet_df['source_id'] = conceptnet_df['source'].astype('category').cat.codes
conceptnet_df['target_id'] = conceptnet_df['target'].astype('category').cat.codes
conceptnet_df['relation_id'] = conceptnet_df['relation'].astype('category').cat.codes


def get_nodes_and_relations(conceptnet_df):
    source_df = conceptnet_df[['source', 'source_id']].rename(columns={'source': 'name', 'source_id': 'id'}).drop_duplicates()
    target_df = conceptnet_df[['target', 'target_id']].rename(columns={'target': 'name', 'target_id': 'id'}).drop_duplicates()
    concept_df = pd.concat([source_df, target_df], ignore_index=True).drop_duplicates().set_index('id')
    relation_df = conceptnet_df[['relation', 'relation_id']].rename(columns={'relation': 'name', 'relation_id': 'id'}).drop_duplicates().set_index('id')
    return concept_df, relation_df


concept_df, relation_df = get_nodes_and_relations(conceptnet_df)


try:
    concept_embedding_path = osp.join(settings.data_dir, 'conceptnet', 'concept_embedding.pkl')
    relation_embedding_path = osp.join(settings.data_dir, 'conceptnet', 'relation_embedding.pkl')

    concept_embedding = pickle.load(open(concept_embedding_path, 'rb'))
    relation_embedding = pickle.load(open(relation_embedding_path, 'rb'))
except:
    # Encoding concepts
    ic('Encoding concepts')
    concept_embedding = bart(concept_df['name'].tolist(), verbose=True)
    concept_embedding = dict(zip(concept_df['name'].tolist(), concept_embedding))
    # Save concept encodings
    with open(concept_embedding_path, 'wb') as f:
        pickle.dump(concept_embedding, f)

    # Encoding relations
    ic('Encoding relations')
    relation_embedding = bart(relation_df['name'].tolist(), verbose=True)
    relation_embedding = dict(zip(relation_df['name'].tolist(), relation_embedding))
    # Save relation encodings
    with open(relation_embedding_path, 'wb') as f:
        pickle.dump(relation_embedding, f)

def triple_ids_to_pyg_data(triple_ids_list):
    """
    Convert a list of list of conceptnet triples to pytorch geometric data.
    """
    data_list = []
    for triple_ids in tqdm(triple_ids_list):
        data = HeteroData()

        triples_df = conceptnet_df.loc[triple_ids]

        ic(triples_df)

        # Load nodes
        concepts, relations = get_nodes_and_relations(triples_df)
        mapping = {index: i for i, index in enumerate(concepts.index)}
        data['concept'].num_nodes = len(mapping)
        data['concept'].x = torch.cat([concept_embedding[concept] for concept in concepts['name']], dim=0)

        # Load edges
        for i, relation in enumerate(relations['name']):
            src = [mapping[i] for i in triples_df[triples_df['relation'] == relation]['source_id'].tolist()]
            dst = [mapping[i] for i in triples_df[triples_df['relation'] == relation]['target_id'].tolist()]
            data['concept', relation, 'concept'].edge_index = torch.tensor([src, dst])
            data['concept', relation, 'concept'].edge_label = relation_embedding[relation].repeat(len(src), 1)
            # Ignoring weight for now

        data_list.append(data)
    return data_list
