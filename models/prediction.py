import torch
from torch import nn

from utils.settings import settings


class Predictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lin = nn.Linear(2 * settings.sent_dim, 3)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, nles):
        embeddings = torch.cat([inputs['Sentences_embedding'], nles], dim=1)
        outputs = self.lin(embeddings)
        loss = self.loss_fn(outputs, inputs['gold_label'])

        return outputs, loss
