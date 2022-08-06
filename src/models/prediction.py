import torch
from torch import nn

from src.settings import settings


class Predictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        in_channels = settings.sent_dim if settings.nle_pred else 2 * settings.sent_dim
        self.lin = nn.Linear(in_channels, 3)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, nles):
        embeddings = nles if settings.nle_pred else torch.cat([inputs['Sentences_embedding'].to(settings.device), nles], dim=1)
        outputs = self.lin(embeddings)
        loss = self.loss_fn(outputs, inputs['gold_label'].to(settings.device))

        return outputs, loss
