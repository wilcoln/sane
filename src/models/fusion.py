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
        self.hidden_dim = settings.hidden_dim // settings.num_attn_heads
        self.k_proj_list = nn.ModuleList([
            nn.Linear(3 * settings.hidden_dim, self.hidden_dim)
            for _ in range(settings.num_attn_heads)
        ])
        self.s_proj_list = nn.ModuleList([
            nn.Linear(settings.hidden_dim, self.hidden_dim)
            for _ in range(settings.num_attn_heads)
        ])

    def multi_head_attention(self, sentences, knowledge):
        attention_list = []
        fused_knowledge_list = []
        for i in range(settings.num_attn_heads):
            S = self.s_proj_list[i](sentences)  # (batch_size, hidden_dim)
            K = self.k_proj_list[i](knowledge)  # (knowledge_size, hidden_dim)
            alignment = S @ K.T / torch.sqrt(self.hidden_dim)  # (batch_size, knowledge_size)
            attention = torch.softmax(alignment, dim=0)  # (batch_size, hidden_dim)
            fused_knowledge = attention @ knowledge  # (batch_size, hidden_dim)
            attention_list.append(attention)  # (batch_size, hidden_dim)
            fused_knowledge_list.append(fused_knowledge)  # (batch_size, hidden_dim)

        # (num_heads, batch_size, knowledge_size), (batch_size, hidden_dim)
        return torch.cat(attention_list, dim=0), torch.cat(fused_knowledge_list, dim=1)

    def forward(self, inputs, knowledge):
        # Project sentences and knowledge to hidden space
        sentences = inputs['Sentences_embedding'].to(settings.device)
        attentions, knowledge = self.multi_head_attention(sentences, knowledge)
        if self.training:
            # Return just the fused knowledge
            return FuserOutput(fused=knowledge)
        else:
            # Return attention scores too
            return FuserOutput(attentions=attentions, fused=knowledge)
