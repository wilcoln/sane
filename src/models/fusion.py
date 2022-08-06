from dataclasses import dataclass

import torch
from torch import nn

from src.settings import settings


@dataclass
class FuserOutput:
    attentions: torch.Tensor
    knowledge: torch.Tensor


class Fuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(3 * settings.hidden_dim, settings.hidden_dim)
        self.s_proj = nn.Linear(settings.sent_dim, settings.hidden_dim)

    def forward(self, inputs, knowledge):
        # Project sentences
        S = self.s_proj(inputs['Sentences_embedding'].to(settings.device))
        K = self.k_proj(knowledge)
        align_scores = S @ K.T
        attentions = torch.softmax(align_scores, dim=0)
        knowledge = attentions @ knowledge
        return FuserOutput(attentions=attentions, knowledge=knowledge)
