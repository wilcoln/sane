from dataclasses import dataclass

import torch
from torch import nn

from src.settings import settings


@dataclass
class FuserOutput:
    fused: torch.Tensor
    attentions: torch.Tensor = None


class Fuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(3 * settings.hidden_dim, settings.hidden_dim)
        self.s_proj = nn.Linear(settings.sent_dim, settings.hidden_dim)

    def forward(self, inputs, knowledge):
        # Project sentences and knowledge to hidden space
        S = self.s_proj(inputs['Sentences_embedding'].to(settings.device))  # (batch_size, hidden_dim)
        K = self.k_proj(knowledge)  # (knowledge_size, hidden_dim)
        align_scores = S @ K.T  # (batch_size, knowledge_size)
        attentions = torch.softmax(align_scores, dim=0)  # (batch_size, knowledge_size)
        knowledge = attentions @ knowledge  # (batch_size, hidden_dim)
        if self.training:
            # Return just the fused knowledge
            return FuserOutput(fused=knowledge)
        else:
            # Return attention scores too
            return FuserOutput(attentions=attentions, fused=knowledge)
