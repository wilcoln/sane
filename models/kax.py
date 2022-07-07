from torch import nn

from models.knowledge_grounding import KnowledgeGrounder
from models.natural_language_explanation import NaturalLanguageExplainer
from models.prediction import Predictor
from models.rationale_extraction import RationaleExtractor
from utils.settings import settings


class KAX(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.rationale_extractor = RationaleExtractor(*args, **kwargs)
        self.knowledge_grounder = KnowledgeGrounder(*args, **kwargs)
        self.natural_language_explainer = NaturalLanguageExplainer(*args, **kwargs)
        self.predictor = Predictor(*args, **kwargs)

    def forward(self, inputs):

        rationales, re_loss = self.rationale_extractor(inputs)
        knowledge_snippets, kg_loss = self.knowledge_grounder(rationales)
        nles, nle_loss = self.natural_language_explainer(inputs, knowledge_snippets)
        outputs, task_loss = self.predictor(inputs, nles)

        return (nles, outputs), settings.alpha*(re_loss + kg_loss) + (1 - settings.alpha)*(nle_loss + task_loss)
