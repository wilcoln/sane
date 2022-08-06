from dataclasses import dataclass

import torch
from torch import nn

from src.settings import settings
from src.utils.embeddings import transformer_mean_pool


@dataclass
class PredictorOutput:
    loss: torch.Tensor
    logits: torch.Tensor


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = settings.sent_dim if settings.nle_pred else 2*settings.sent_dim
        self.lin = nn.Linear(in_channels, 3)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, nle):
        nle_embed = transformer_mean_pool(nle.last_hidden_state)
        sent_embed = inputs['Sentences_embedding'].to(settings.device)
        input_pred = nle_embed if settings.nle_pred else torch.cat([sent_embed, nle_embed], dim=1)
        logits = self.lin(input_pred)
        loss = self.loss_fn(logits, inputs['gold_label'].to(settings.device))

        return PredictorOutput(logits=logits, loss=loss)
