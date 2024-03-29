from torch import nn

from src.models.attention import Attention
from src.models.encoder import Encoder
from src.models.explanation import Explainer, ExplainerNoKnowledge
from src.models.prediction import Predictor, PredictorNoKnowledge


class SANE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.attention = Attention()
        self.explainer = Explainer()
        self.predictor = Predictor()

    def forward(self, inputs):
        knwl = self.encoder(inputs)
        att_knwl = self.attention(inputs, knwl)
        nle = self.explainer(inputs, att_knwl.output)
        pred = self.predictor(inputs, att_knwl.output, nle)
        return pred, nle, att_knwl, knwl

    @property
    def g_modules(self):
        return {
            self.encoder,
            self.attention,
            self.explainer.model.model.g1,
            self.predictor.g1,
        }

    @property
    def h_modules(self):
        return {
            self.explainer.model.model.encoder,
            self.predictor.h,
        }

    @property
    def f_modules(self):
        return {
            self.explainer.model.model.decoder,
            self.explainer.model.lm_head,
            self.predictor.f,
        }


class SANENoKnowledge(nn.Module):
    def __init__(self):
        super().__init__()
        self.explainer = ExplainerNoKnowledge()
        self.predictor = PredictorNoKnowledge()

    def forward(self, inputs):
        nle = self.explainer(inputs)
        pred = self.predictor(inputs, nle)
        return pred, nle
