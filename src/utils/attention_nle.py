import math
from dataclasses import dataclass

import torch
from torch import nn

from src.settings import settings


@dataclass
class AttentionOutput:
    output: torch.Tensor
    weights: torch.Tensor = None


class AttentionNLE(nn.Module):
    def __init__(self):
        super().__init__()

        assert settings.hidden_dim % settings.num_attn_heads == 0, 'Hidden dim must be divisible by num attention heads'
        self.hidden_dim = settings.hidden_dim // settings.num_attn_heads
        self.k_proj_list = nn.ModuleList([
            nn.Linear(settings.sent_dim, self.hidden_dim)
            for _ in range(settings.num_attn_heads)
        ])
        self.s_proj_list = nn.ModuleList([
            nn.Linear(settings.sent_dim, self.hidden_dim)
            for _ in range(settings.num_attn_heads)
        ])

        self.transform = nn.Sequential(
            nn.Linear(settings.num_attn_heads * settings.sent_dim, settings.sent_dim),
            nn.ReLU(),
            nn.Linear(settings.sent_dim, settings.sent_dim),
        )

    def forward(self, encoder_last_hidden_state, knowledge):
        # compute attention weights & output for each attention head
        attention_weights_list = []
        attention_output_list = []
        for i in range(settings.num_attn_heads):
            # Project sentences and knowledge
            S = self.s_proj_list[i](encoder_last_hidden_state)  # (batch_size, seq_len, hidden_dim)
            K = self.k_proj_list[i](knowledge)  # (knowledge_size, hidden_dim)
            # scaled dot product attention
            alignment_weights = S @ K.T / math.sqrt(self.hidden_dim)  # (batch_size, seq_len, knowledge_size)
            attention_weights = torch.softmax(alignment_weights, dim=2)  # (batch_size, seq_len, knowledge_size)
            attention_output = attention_weights @ knowledge  # (batch_size, seq_len, sent_dim)
            # save attention head weights and outputs
            attention_weights_list.append(attention_weights)  # (batch_size, seq_len, sent_dim)
            attention_output_list.append(attention_output)  # (batch_size, seq_len, sent_dim)

        # (num_heads, batch_size, seq_len, knowledge_size), (batch_size, seq_len, sent_dim)
        weights, output = torch.cat(attention_weights_list, dim=0), self.transform(
            torch.cat(attention_output_list, dim=1))

        return AttentionOutput(output=output)
