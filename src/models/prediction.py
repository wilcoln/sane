from dataclasses import dataclass

import torch
from torch import nn

from src.settings import settings
from src.utils.embeddings import transformer_sentence_pool


@dataclass
class PredictorOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    logits_no_knowledge: torch.Tensor
    knowledge_relevance: torch.Tensor = None
    loss_no_knowledge: torch.Tensor = None


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2 * settings.sent_dim, settings.num_classes)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.fusion_head = nn.Linear(2 * settings.sent_dim, settings.sent_dim)
        self.transform = nn.Sequential(
            nn.Linear(settings.sent_dim, settings.sent_dim),
            nn.ReLU(),
            nn.Linear(settings.sent_dim, settings.sent_dim),
        )

    def forward(self, inputs, knwl, nle):
        sent_embed, labels = inputs['Sentences_embedding'].to(settings.device), inputs['gold_label'].to(settings.device)
        sent_embed = self.transform(sent_embed)

        # With knowledge
        nle_embed = transformer_sentence_pool(nle.last_hidden_state)
        r = torch.sigmoid(self.fusion_head(torch.cat([sent_embed, knwl], dim=1)))
        input_pred = torch.cat([sent_embed + r * knwl, nle_embed], dim=1)
        logits = self.lin(input_pred)
        loss = self.loss_fn(logits, labels)
        # Without knowledge
        nle_embed = transformer_sentence_pool(nle.last_hidden_state_no_knowledge)
        input_pred_nk = torch.cat([sent_embed, nle_embed], dim=1)
        logits_nk = self.lin(input_pred_nk)
        loss_nk = self.loss_fn(logits_nk, labels)

        return PredictorOutput(logits=logits, logits_no_knowledge=logits_nk, loss=loss, loss_no_knowledge=loss_nk, knowledge_relevance=r)


class PredictorNoKnowledge(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2 * settings.sent_dim, settings.num_classes)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, nle):
        nle_embed = transformer_sentence_pool(nle.last_hidden_state)
        sent_embed = inputs['Sentences_embedding'].to(settings.device)
        input_pred = torch.cat([sent_embed, nle_embed], dim=1)
        logits = self.lin(input_pred)
        loss = self.loss_fn(logits, inputs['gold_label'].to(settings.device))

        return PredictorOutput(logits=logits, loss=loss)

