import torch
from icecream import ic
from torch import nn
from utils.settings import settings
loss_fn = nn.CrossEntropyLoss()
from utils.nn import MLP


class Predictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mlp = MLP(in_channels=384*2, out_channels=3, hidden_channels=settings.hidden_dim)

    def forward(self, inputs, nles):
        embeddings = ic(torch.cat([inputs['Sentences_embeddings'], nles], dim=1))
        outputs = self.mlp(embeddings)
        loss = loss_fn(outputs, inputs['gold_label'])

        return outputs, loss
