from torch import nn

from src.settings import settings
from src.utils.transformers import BartForKnowledgeAwareConditionalGeneration, BartForConditionalGeneration


class Explainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BartForKnowledgeAwareConditionalGeneration.from_pretrained("facebook/bart-base")

    def forward(self, inputs, knowledge):
        # send tensors to gpu
        encoded_inputs = {k: v.to(settings.device) for k, v in inputs['Sentences'].items()}
        encoded_labels = {k: v.to(settings.device) for k, v in inputs['Explanation_1'].items()}
        encoded_knowledge = {'knowledge_embedding': knowledge}
        return self.model(**encoded_inputs, **encoded_knowledge, labels=encoded_labels['input_ids'])


class ExplainerNoKnowledge(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    def forward(self, inputs):
        # send tensors to gpu
        encoded_inputs = {k: v.to(settings.device) for k, v in inputs['Sentences'].items()}
        encoded_labels = {k: v.to(settings.device) for k, v in inputs['Explanation_1'].items()}
        return self.model(**encoded_inputs, labels=encoded_labels['input_ids'])
