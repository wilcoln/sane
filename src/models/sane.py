import torch
from torch import nn

from src.models.attention import Attention
from src.models.explanation import Explainer
from src.models.knowledge import Encoder
from src.models.prediction import Predictor

torch.manual_seed(0)


class SANE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.attention = Attention()
        self.explainer = Explainer()
        self.predictor = Predictor()

    def forward(self, inputs):
        knwl = self.encoder(inputs)
        att_knwl = self.attention(inputs, knwl.embed)
        nle = self.explainer(inputs, att_knwl.output)
        pred = self.predictor(inputs, nle)
        return knwl, att_knwl, nle, pred
