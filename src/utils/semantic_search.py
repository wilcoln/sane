from typing import List

import torch

from src.utils.embeddings import bart


def semantic_search(queries: List[str], values: List[str], top_k=10, encode=False) -> torch.Tensor:
    if encode:
        queries, values = bart(queries), bart(values)

    return torch.topk(queries @ values.T, min(top_k, len(values)), dim=1)[1]


if __name__ == '__main__':
    queries_ = ['brown_hair', 'sports_field', 'hug', 'softball', 'package', 'black', 'white', 'hiking', 'mountain']
    values_ = [
        'The sisters are hugging goodbye while holding to go packages after just eating lunch.',
        'A female softball player wearing blue and red crouches in the infield, waiting for the next play.',
        'Two girls kissing a man with a black shirt and brown hair on the cheeks',
        'A woman with a black and white dress on is carrying a green plastic laundry basket',
        'Two lacrosse players are running on the sports-field. One of the boys scored a goal',
        'Three people hiking up a mountain with a river and other mountains in the background.',
    ]
    print(semantic_search(queries_, values_, top_k=5, encode=True))
