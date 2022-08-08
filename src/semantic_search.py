from typing import List

import torch
from sentence_transformers import SentenceTransformer, util as sbert_util
from src.utils.embeddings import bart

def semantic_search(queries: List[str], values: List[str], top_k=10, encode=False) -> torch.Tensor:
    
    if encode:
        queries = bart(queries)
        values = bart(values)

    return torch.topk(queries @ values.T, min(top_k, len(values)), dim=1)[1]


if __name__ == '__main__':
    
    query = ['brown_hair', 'sports_field', 'hug', 'softball', 'package', 'black', 'white', 'hiking', 'mountain']
    values = [
        'The sisters are hugging goodbye while holding to go packages after just eating lunch.',
        'A female softball player wearing blue and red crouches in the infield, waiting for the next play.',
        'Two girls kissing a man with a black shirt and brown hair on the cheeks',
        'A woman with a black and white dress on is carrying a green plastic laundry basket in front of an apartment building',
        'Two lacrosse players are running on the sports-field. One of the boys scored a goal for his sports team today.',
        'Three people hiking up a mountain with a river and other mountains in the background. Hikers in the mountains.',
    ]
    print(semantic_search(query, values, top_k=5, encode=True))
