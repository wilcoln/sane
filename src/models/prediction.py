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
    knowledge_contribution: torch.Tensor = None
    floss: torch.Tensor = None


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(2 * settings.sent_dim, settings.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(settings.hidden_dim, settings.num_classes),
        )
        self.g1 = nn.Sequential(
            nn.Linear(2 * settings.sent_dim, settings.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(settings.hidden_dim, 1),
            nn.Sigmoid()
        )
        self.h = nn.Sequential(
            nn.Linear(settings.sent_dim, settings.sent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(settings.sent_dim, settings.sent_dim),
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, knwl, nle):
        sent_embed, labels = inputs['Sentences_embedding'].to(settings.device), inputs['gold_label'].to(settings.device)
        nle_embed = transformer_sentence_pool(nle.last_hidden_state)
        x_tilde = self.h(sent_embed)

        # Knowledge relevance
        r = self.g1(torch.hstack((sent_embed, knwl)))

        # Knowledge integration
        rk_tilde = r * knwl
        sent_embed = x_tilde + rk_tilde

        # Knowledge contribution
        ck = torch.norm(rk_tilde, dim=1) ** 2
        cx = torch.norm(x_tilde, dim=1) ** 2
        c = ck / (ck + cx)

        # Prediction
        input_pred = torch.hstack((sent_embed, nle_embed))
        logits = self.f(input_pred)
        loss = self.loss_fn(logits, labels)

        return PredictorOutput(logits=logits, loss=loss, knowledge_relevance=r, knowledge_contribution=c)


class PredictorNoKnowledge(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(settings.sent_dim, settings.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(settings.hidden_dim, settings.num_classes),
        )
        self.h = nn.Sequential(
            nn.Linear(settings.sent_dim, settings.sent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(settings.sent_dim, settings.sent_dim),
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, nle):
        sent_embed, labels = inputs['Sentences_embedding'].to(settings.device), inputs['gold_label'].to(settings.device)
        nle_embed = transformer_sentence_pool(nle.last_hidden_state)

        sent_embed = self.h(sent_embed)

        # Prediction
        floss = torch.norm(sent_embed - nle_embed, dim=1) ** 2
        logits = self.f(sent_embed)
        loss = self.loss_fn(logits, labels)

        return PredictorOutput(logits=logits, loss=loss, floss=floss)
