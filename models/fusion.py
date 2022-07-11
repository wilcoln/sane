from torch import nn

from utils.embeddings import bert


class Fuser(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs):
        inputs['Sentence1_embeddings'] = bert(inputs['Sentence1'], verbose=False)
        inputs['Sentence2_embeddings'] = bert(inputs['Sentence2'], verbose=False)
        return inputs
