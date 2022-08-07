import torch
from torch import nn

from src.models.encode import Encoder
from src.models.explanation import Explainer
from src.models.fusion import Fuser
from src.models.prediction import Predictor
torch.manual_seed(0)


class KAX(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.fuser = Fuser()
        self.explainer = Explainer()
        self.predictor = Predictor()

    def forward(self, inputs):
        knwl = self.encoder(inputs)
        fused_knwl = self.fuser(inputs, knwl.encoded)
        nle = self.explainer(inputs, fused_knwl.fused)
        pred = self.predictor(inputs, nle)
        return knwl, fused_knwl, nle, pred
