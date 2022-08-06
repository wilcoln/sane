import torch
from torch import nn

from src.settings import settings


class Fuser(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.k_proj = nn.Linear(3 * settings.hidden_dim, settings.hidden_dim)
        self.s_proj = nn.Linear(settings.sent_dim, settings.hidden_dim)

    def forward(self, inputs):
        # Project sentences
        S = self.s_proj(inputs['Sentences_embedding'].to(settings.device))
        K = self.k_proj(inputs['Knowledge_embedding'])
        V = inputs['Knowledge_embedding']
        align_scores = S @ K.T
        attn_scores = torch.softmax(align_scores, dim=0)
        inputs['Knowledge_embedding'] = attn_scores @ V
        return attn_scores, inputs
