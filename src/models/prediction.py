from dataclasses import dataclass

import torch
from torch import nn

from src.settings import settings
from src.utils.embeddings import transformer_sentence_pool


@dataclass
class PredictorOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    knowledge_relevance: torch.Tensor = None


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2 * settings.sent_dim, settings.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.fusion_head = nn.Linear(2 * settings.sent_dim, settings.sent_dim)

    def forward(self, inputs, knwl, nle):
        nle_embed = transformer_sentence_pool(nle.last_hidden_state)
        sent_embed = inputs['Sentences_embedding'].to(settings.device)
        r = torch.sigmoid(self.fusion_head(torch.cat([sent_embed, knwl], dim=1)))
        sent_embed = sent_embed + r * knwl
        input_pred = torch.cat([sent_embed, nle_embed], dim=1)
        logits = self.lin(input_pred)
        loss = self.loss_fn(logits, inputs['gold_label'].to(settings.device))

        return PredictorOutput(logits=logits, loss=loss, knowledge_relevance=r)


class PredictorNoKnowledge(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2 * settings.sent_dim, settings.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, nle):
        nle_embed = transformer_sentence_pool(nle.last_hidden_state)
        sent_embed = inputs['Sentences_embedding'].to(settings.device)
        input_pred = torch.cat([sent_embed, nle_embed], dim=1)
        logits = self.lin(input_pred)
        loss = self.loss_fn(logits, inputs['gold_label'].to(settings.device))

        return PredictorOutput(logits=logits, loss=loss)
