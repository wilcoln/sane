from typing import List

import torch
from sentence_transformers import SentenceTransformer, util as sbert_util


def semantic_search(query_list: List[str], corpus_list: List[str], top_k=10) -> torch.Tensor:
    model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    # Two lists of sentences

    # Compute embedding for both lists
    query_embeddings = model.encode(query_list, convert_to_tensor=True, batch_size=32)
    corpus_embeddings = model.encode(corpus_list, convert_to_tensor=True, batch_size=32)

    return sbert_util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k, corpus_chunk_size=500)


if __name__ == '__main__':
    query = ['The dog is running', 'The cat is running']
    corpus = ['The dog is running', 'The cat is running', 'The dog is eating', 'The cat is eating']
    print(semantic_search(query, corpus))
