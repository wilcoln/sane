from icecream import ic
from torch import nn

from models.fusion import Fuser
from models.explanation import Explainer
from models.prediction import Predictor
from utils.settings import settings


class KAX(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fuser = Fuser(*args, **kwargs)
        self.explainer = Explainer(*args, **kwargs)
        self.predictor = Predictor(*args, **kwargs)

    def forward(self, inputs):
        # fuse two modalities
        ic('fusing')
        inputs = self.fuser(inputs)
        ic('explaining')
        nles, nle_loss = self.explainer(inputs)
        # fuse explanation and inputs (orthogonal ?)
        ic('predicting')
        outputs, task_loss = self.predictor(inputs, nles)
        ic('done')
        return nles, outputs, settings.alpha*nle_loss + (1 - settings.alpha)*task_loss
