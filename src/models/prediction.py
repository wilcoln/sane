from dataclasses import dataclass

import torch
from torch import nn

from src.models.attention import Attention
from src.settings import settings
from src.utils.embeddings import transformer_sentence_pool


@dataclass
class PredictorOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    knowledge_relevance: torch.Tensor = None
    knowledge_attention_weights: torch.Tensor = None


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.pred_head = nn.Linear(2 * settings.sent_dim, settings.num_classes)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.fusion_head = nn.Linear(2 * settings.sent_dim, 1)
        self.transform = nn.Sequential(
            nn.Linear(settings.sent_dim, settings.sent_dim),
            nn.ReLU(),
            nn.Linear(settings.sent_dim, settings.sent_dim),
        )

    def forward(self, inputs, knwl, nle):
        # Knowledge attention
        att_knwl = self.attention(inputs, knwl.output)

        # Send tensors to GPU
        sent_embed, labels = inputs['Sentences_embedding'].to(settings.device), inputs['gold_label'].to(settings.device)
        # Average nle token embeddings
        nle_embed = transformer_sentence_pool(nle.last_hidden_state)
        # Compute knowledge relevance
        r = torch.sigmoid(self.fusion_head(torch.cat([sent_embed, att_knwl.output], dim=1)))
        # transform input sentence embedding
        sent_embed = self.transform(sent_embed)
        # Augment sentence embedding with knowledge and nle
        input_pred = torch.cat([sent_embed + r * att_knwl.output, nle_embed], dim=1)
        logits = self.pred_head(input_pred)
        loss = self.loss_fn(logits, labels)

        return PredictorOutput(loss=loss, logits=logits, knowledge_relevance=r, knowledge_attention_weights=att_knwl.weights)


class PredictorNoKnowledge(nn.Module):
    def __init__(self):
        super().__init__()
        self.pred_head = nn.Linear(2 * settings.sent_dim, settings.num_classes)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, nle):
        sent_embed, labels = inputs['Sentences_embedding'].to(settings.device), inputs['gold_label'].to(settings.device)
        # Average nle token embeddings
        nle_embed = transformer_sentence_pool(nle.last_hidden_state)
        # Augment sentence embedding with and nle
        input_pred = torch.cat([sent_embed, nle_embed], dim=1)
        logits = self.pred_head(input_pred)
        loss = self.loss_fn(logits, labels)

        return PredictorOutput(logits=logits, loss=loss)

