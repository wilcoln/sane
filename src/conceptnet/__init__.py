"""
Requirements:
- Given a text, return a list of candidate concepts.
- Given a text, return a list of candidate relation types.
- Given a sentence, get it's associated subknowledge graph of up to k hops, i.e. all concepts appearing in the sentence.
- Be able to merge subgraphs.
- Get the embedding of concepts
"""
import os.path as osp
from typing import Tuple, List

import networkx as onx
import pandas as pd

from src.utils.nltk import all_grams
from src.utils.settings import settings


class Conceptnet:
    nx = None
    df = None

    def __init__(self):
        cn_dir = osp.join(settings.data_dir, 'conceptnet')
        csv_path = osp.join(cn_dir, 'conceptnet-assertions-5.6.0_cleaned.csv')
        self.df = pd.read_csv(csv_path)
        for column in {'source', 'target', 'relation'}:
            self.df[column] = self.df[column].astype('str')
        self.concept_df, self.relation_df = self.decompose_df(self.df)
        self.concept_dict = dict(zip(self.concept_df['name'], self.concept_df.index))
        self.relation_dict = dict(zip(self.relation_df['name'], self.relation_df.index))
        self.inv_concept_dict = dict(zip(self.concept_df.index, self.concept_df['name']))
        self.inv_relation_dict = dict(zip(self.relation_df.index, self.relation_df['name']))

        if self.nx is None:
            gpickle_path = osp.join(cn_dir, 'conceptnet.gpickle')
            try:
                self.nx = onx.read_gpickle(gpickle_path)
            except FileNotFoundError:
                self.nx = onx.from_pandas_edgelist(
                    self.df, source='source', target='target', edge_key='relation',
                    edge_attr='weight', create_using=onx.MultiDiGraph())
                onx.write_gpickle(self.nx, gpickle_path)

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

    def search(self, query, limit=None) -> List[str]:
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

    def nodes2ids(self, nodes: list) -> list:
        return [self.concept_dict[node] for node in nodes]

    @staticmethod
    def decompose_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        source_df = df[['source']].rename(columns={'source': 'name'}).drop_duplicates()
        target_df = df[['target']].rename(columns={'target': 'name'}).drop_duplicates()
        concept_df = pd.concat([source_df, target_df], ignore_index=True).drop_duplicates().reset_index()
        relation_df = df[['relation']].rename(columns={'relation': 'name'}).drop_duplicates().reset_index()
        return concept_df, relation_df

