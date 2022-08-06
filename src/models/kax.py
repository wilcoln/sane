from torch import nn

from src.models.encode import Encoder
from src.models.explanation import Explainer
from src.models.fusion import Fuser
from src.models.prediction import Predictor


class KAX(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = Encoder(*args, **kwargs)
        self.fuser = Fuser(*args, **kwargs)
        self.explainer = Explainer(*args, **kwargs)
        self.predictor = Predictor(*args, **kwargs)

    def forward(self, inputs):
        knwl = self.encoder(inputs)
        att_knwl = self.fuser(inputs, knwl)
        nle = self.explainer(inputs, att_knwl.knowledge)
        pred = self.predictor(inputs, nle)
        return att_knwl, nle, pred
