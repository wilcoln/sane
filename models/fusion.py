import pandas as pd
from torch import nn

import os.path as osp
from utils.embeddings import bart
from utils.settings import settings


class Fuser(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs):
        if 'Sentences_embeddings' not in inputs:
            inputs['Sentences'] = [f'{sent1} -> {sent2}' for sent1, sent2 in zip(inputs['Sentence1'], inputs['Sentence2'])]
            inputs['Sentences_embeddings'] = bart(inputs['Sentences'])

        return inputs
