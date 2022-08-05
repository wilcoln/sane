from torch import nn

from src.models.explanation import Explainer, ExplainerWithoutKnowledge
from src.models.fusion import Fuser, Encoder
from src.models.prediction import Predictor
from src.utils.settings import settings


class KAX(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = Encoder(*args, **kwargs)
        self.fuser = Fuser(*args, **kwargs)
        self.explainer = Explainer(*args, **kwargs)
        self.predictor = Predictor(*args, **kwargs)

    def forward(self, inputs):
        # fuse two modalities
        # ic('encode')
        # start = time.time()
        inputs = self.encoder(inputs)
        # ic(f'done in {time.time() - start}')
        # ic('fusing')
        # start = time.time()
        attn_scores, inputs = self.fuser(inputs)
        # ic(f'done in {time.time() - start}')
        # ic('explaining')
        # start = time.time()
        nles_tokens, nles, nle_loss = self.explainer(inputs)
        # ic(f'done in {time.time() - start}')
        # fuse explanation and inputs (orthogonal ?)
        # ic('predicting')
        # start = time.time()
        outputs, task_loss = self.predictor(inputs, nles)
        # ic(f'done in {time.time() - start}')
        return attn_scores, nles_tokens, nles, outputs, settings.alpha * nle_loss + (1 - settings.alpha) * task_loss


class KAXWK(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.explainer = ExplainerWithoutKnowledge(*args, **kwargs)
        self.predictor = Predictor(*args, **kwargs)

    def forward(self, inputs):
        nles_tokens, nles, nle_loss = self.explainer(inputs)
        # ic(f'done in {time.time() - start}')
        # fuse explanation and inputs (orthogonal ?)
        # ic('predicting')
        # start = time.time()
        outputs, task_loss = self.predictor(inputs, nles)
        # ic(f'done in {time.time() - start}')
        return None, nles_tokens, nles, outputs, settings.alpha * nle_loss + (1 - settings.alpha) * task_loss
