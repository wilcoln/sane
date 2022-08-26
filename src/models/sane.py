from torch import nn

from src.models.explanation import Explainer, ExplainerNoKnowledge
from src.models.encoder import Encoder
from src.models.prediction import Predictor, PredictorNoKnowledge


class SANE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.explainer = Explainer()
        self.predictor = Predictor()

    def forward(self, inputs):
        knwl = self.encoder(inputs)
        nle = self.explainer(inputs, knwl.output)
        pred = self.predictor(inputs, knwl.output, nle)
        return pred, nle, knwl

    @property
    def g_modules(self):
        return {
            self.encoder,
            self.attention,
            self.explainer.model.model.fusion_head,
            self.predictor.fusion_head,
        }

    @property
    def h_modules(self):
        return {
            self.explainer.model.model.encoder,
            self.explainer.model.model.transform,
            self.predictor.transform,
        }

    @property
    def f_modules(self):
        return {
            self.explainer.model.model.decoder,
            self.explainer.model.lm_head,
            self.predictor.pred_head,
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
