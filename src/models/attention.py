from dataclasses import dataclass
import math
import torch
from torch import nn

from src.settings import settings


@dataclass
class AttentionOutput:
    output: torch.Tensor
    weights: torch.Tensor = None


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        assert settings.hidden_dim % settings.num_attn_heads == 0, 'Hidden dimension must be divisible by num attention heads'
        self.hidden_dim = settings.hidden_dim // settings.num_attn_heads
        self.k_proj_list = nn.ModuleList([
            nn.Linear(settings.sent_dim, self.hidden_dim)
            for _ in range(settings.num_attn_heads)
        ])
        self.s_proj_list = nn.ModuleList([
            nn.Linear(settings.sent_dim, self.hidden_dim)
            for _ in range(settings.num_attn_heads)
        ])

        self.final_proj = nn.Linear(settings.num_attn_heads * settings.sent_dim, settings.sent_dim)

    def forward(self, inputs, knowledge):
        # send tensors to gpu
        sentences = inputs['Sentences_embedding'].to(settings.device)

        # compute attention weights & output for each attention head
        attention_weights_list = []
        attention_output_list = []
        for i in range(settings.num_attn_heads):
            # Project sentences and knowledge
            S = self.s_proj_list[i](sentences)  # (batch_size, hidden_dim)
            K = self.k_proj_list[i](knowledge)  # (knowledge_size, hidden_dim)
            # scaled dot product attention
            alignment_weights = S @ K.T / math.sqrt(self.hidden_dim)  # (batch_size, knowledge_size)
            attention_weights = torch.softmax(alignment_weights, dim=0)  # (batch_size, hidden_dim)
            attention_output = attention_weights @ knowledge  # (batch_size, hidden_dim)
            # save attention head weights and outputs
            attention_weights_list.append(attention_weights)  # (batch_size, hidden_dim)
            attention_output_list.append(attention_output)  # (batch_size, hidden_dim)

        # (num_heads, batch_size, knowledge_size), (batch_size, hidden_dim)
        weights, output = torch.cat(attention_weights_list, dim=0), self.final_proj(torch.cat(attention_output_list, dim=1))

        if self.training:
            # Return just the attention output during training
            return AttentionOutput(output=output)
        else:
            # Return attention weights too
            return AttentionOutput(weights=weights, output=output)