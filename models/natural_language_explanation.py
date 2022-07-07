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
        self.mlp = MLP(in_channels=384*3, out_channels=3, hidden_channels=settings.hidden_dim)

    def forward(self, inputs, nles):

        # Encode premise and hypothesis
        premises_embeddings = inputs['Sentence1']
        hypothesis_embeddings = inputs['Sentence2']

        # premises_embeddings = bert(premises, verbose=False)
        # hypothesis_embeddings = bert(hypothesis, verbose=False)

        outputs = self.mlp(torch.cat([premises_embeddings, hypothesis_embeddings, nles], dim=1))

        loss = loss_fn(outputs, inputs['gold_label'])

        return outputs, loss
