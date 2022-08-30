from torch import nn

from src.settings import settings
from src.utils.bart import BartForConditionalGeneration
from src.utils.bart_with_knowledge import BartWithKnowledgeForConditionalGeneration
from src.utils.embeddings import frozen_bart_model


class Explainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BartWithKnowledgeForConditionalGeneration.from_pretrained("facebook/bart-base")

    def forward(self, inputs, knowledge):
        # send tensors to gpu
        encoded_inputs = {k: v.to(settings.device) for k, v in inputs['Sentences'].items()}
        encoded_labels = {k: v.to(settings.device) for k, v in inputs['Explanation_1'].items()}
        init_input_embeds = frozen_bart_model(**encoded_inputs).last_hidden_state
        encoded_knowledge = {'knowledge_embedding': knowledge, 'init_input_embeds': init_input_embeds}
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
