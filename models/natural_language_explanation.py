from torch import nn


class NaturalLanguageExplainer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs, knowledge_snippets):
        nles = inputs
        return nles, 0


import torch
from torch import nn
from utils.settings import settings
loss_fn = nn.CrossEntropyLoss()
from utils.embeddings import bert
from utils.nn import MLP


class Predictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, inputs, nles):
        pass
