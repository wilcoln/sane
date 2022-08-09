from torch import nn

from src.models.attention import Attention
from src.models.explanation import Explainer, ExplainerNoKnowledge
from src.models.knowledge import Encoder
from src.models.prediction import Predictor


class SANE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.attention = Attention()
        self.explainer = Explainer()
        self.predictor = Predictor()

    def forward(self, inputs):
        knwl = self.encoder(inputs)
        att_knwl = self.attention(inputs, knwl.output)
        nle = self.explainer(inputs, att_knwl.output)
        pred = self.predictor(inputs, nle)
        return knwl, att_knwl, nle, pred


class SANENoKnowledge(nn.Module):
    def __init__(self):
        super().__init__()
        self.explainer = ExplainerNoKnowledge()
        self.predictor = Predictor()

    def forward(self, inputs):
        nle = self.explainer(inputs)
        pred = self.predictor(inputs, nle)
        return nle, pred
