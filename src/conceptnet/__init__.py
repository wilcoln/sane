"""
Requirements:
- Given a text, return a list of candidate concepts.
- Given a text, return a list of candidate relation types.
- Given a sentence, get it's associated subknowledge graph of up to k hops, i.e. all concepts appearing in the sentence.
- Be able to merge subgraphs.
- Get the embedding of concepts
"""
import os.path as osp
import random
from typing import Tuple, List, Union

import networkx as onx
import pandas as pd
import torch
from torch_geometric.data import Data

from src.settings import settings
from src.utils.nltk import all_grams
from src.utils.types import ChunkedList


class Conceptnet:
    def __init__(self):
        self.inv_relation_dict = None
        self.inv_concept_dict = None
        cn_dir = osp.join(settings.data_dir, 'conceptnet')

        # Dataframe objects
        csv_path = osp.join(cn_dir, 'conceptnet-assertions-5.6.0_cleaned.csv')
        self.df = pd.read_csv(csv_path)
        for column in {'source', 'target', 'relation'}:
            self.df[column] = self.df[column].astype('str')
        concept_df, relation_df = self.decompose_df(self.df)

        # Dictionary objects
        self.concept_dict = dict(zip(concept_df['name'], concept_df.index))
        self.relation_dict = dict(zip(relation_df['name'], relation_df.index))
        self.inv_concept_dict = {v: k for k, v in self.concept_dict.items()}
        self.inv_relation_dict = {v: k for k, v in self.relation_dict.items()}

        # Networkx object
        gpickle_path = osp.join(cn_dir, 'conceptnet.gpickle')
        try:
            self.nx = onx.read_gpickle(gpickle_path)
        except FileNotFoundError:
            self.nx = onx.from_pandas_edgelist(
                self.df, source='source', target='target', edge_key='relation',
                edge_attr='weight', create_using=onx.MultiDiGraph())
            onx.write_gpickle(self.nx, gpickle_path)

        # PyG object
        pyg_path = osp.join(cn_dir, 'conceptnet2.pyg')
        try:
            self.pyg = torch.load(pyg_path)
        except FileNotFoundError:
            self.pyg = self.nx2pyg()
            torch.save(self.pyg, pyg_path)

        # Concept Embeddings
        concept_embedding = ChunkedList(n=len(self.concept_dict), dirpath=osp.join(cn_dir, 'concept_embedding'))
        self.concept_embedding = torch.cat(concept_embedding.get_chunks(), dim=0)

        # Relation Embeddings
        relation_embedding = ChunkedList(n=len(self.relation_dict), dirpath=osp.join(cn_dir, 'relation_embedding'))
        self.relation_embedding = torch.cat(relation_embedding.get_chunks(), dim=0)

    @property
    def size(self) -> Tuple[int, int]:
        return self.nx.number_of_nodes(), self.nx.number_of_edges()

    def neighbors(self, nodes: list, radius: int = 0) -> set:
        if radius > 0:
            nodes = set(nodes)
            anchor_nodes = nodes.copy()
            for node in anchor_nodes:
                neighbors = onx.ego_graph(self.nx, node, radius).nodes
                nodes |= set(neighbors)

        return nodes

    def subgraph(self, nodes: list, radius: int = 0) -> onx.MultiDiGraph:
        nodes = self.neighbors(nodes, radius)
        return self.nx.subgraph(nodes)

    def search(self, query: str, limit: int = None) -> List[str]:
        ngrams = all_grams(query)
        count = 0
        results = []
        for ngram in ngrams:
            if ngram in self.concept_dict:
                count += 1
                results.append(ngram)
            if limit is not None and count >= limit:
                break
        return results

    def ids2triples(self, ids: list) -> list:
        return [
            (self.inv_concept_dict[hid], self.inv_relation_dict[rid], self.inv_concept_dict[tid])
            for hid, rid, tid in ids
        ]

    def nodes2ids(self, nodes: list) -> list:
        return [self.concept_dict[node] for node in nodes]

    def sample_neighbors(self, concept: Union[str, int], sample_size: int = 10) -> List[str]:
        if isinstance(concept, int):
            concept = self.inv_concept_dict[concept]
            neighbors = self.nodes2ids(self.nx.neighbors(concept))
        else:
            neighbors = self.nx.neighbors(concept)

        if sample_size < len(neighbors):
            return random.sample(neighbors, sample_size)
        else:
            return neighbors

    @staticmethod
    def decompose_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        source_df = df[['source']].rename(columns={'source': 'name'}).drop_duplicates()
        target_df = df[['target']].rename(columns={'target': 'name'}).drop_duplicates()
        concept_df = pd.concat([source_df, target_df], ignore_index=True).drop_duplicates().reset_index()
        relation_df = df[['relation']].rename(columns={'relation': 'name'}).drop_duplicates().reset_index()
        return concept_df, relation_df

    def nx2pyg(self):
        _nx = onx.relabel_nodes(self.nx, self.concept_dict)

        x = torch.Tensor(list(self.concept_dict.values()))
        edge_index = torch.tensor(list(_nx.edges(keys=False)), dtype=torch.long).t().contiguous().view(2, -1)
        edge_attr = torch.tensor([
            [self.relation_dict[key], data['weight']]
            for u, v, key, data in _nx.edges(keys=True, data=True)
        ], dtype=torch.float).view()

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# Singleton object
conceptnet = Conceptnet()
