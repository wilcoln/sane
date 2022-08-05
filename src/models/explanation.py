from torch import nn
from src.utils.embeddings import transformer_sentence_pool
from src.settings import settings
from src.utils.transformers import BartForExplanationGeneration, BartForExplanationGenerationWK


class Explainer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = BartForExplanationGeneration.from_pretrained("facebook/bart-base")
        self.model.set_lm_head(knowledge_dim=3 * settings.hidden_dim)

    def forward(self, inputs):

        # send tensors to gpu
        encoded_inputs = {k: v.to(settings.device) for k, v in inputs['Sentences'].items()}
        encoded_labels = {k: v.to(settings.device) for k, v in inputs['Explanation_1'].items()}
        encoded_knowledge = {'knowledge_embedding': inputs['Knowledge_embedding']}
        out_tokens, model_outputs = self.model(**encoded_inputs, **encoded_knowledge,
                                               labels=encoded_labels['input_ids'])
        return out_tokens, transformer_sentence_pool(model_outputs, encoded_inputs), model_outputs['loss']


class ExplainerWithoutKnowledge(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = BartForExplanationGenerationWK.from_pretrained("facebook/bart-base")

    def forward(self, inputs):
        # send tensors to gpu
        encoded_inputs = {k: v.to(settings.device) for k, v in inputs['Sentences'].items()}
        encoded_labels = {k: v.to(settings.device) for k, v in inputs['Explanation_1'].items()}
        out_tokens, model_outputs = self.model(**encoded_inputs, labels=encoded_labels['input_ids'])

        return out_tokens, transformer_sentence_pool(model_outputs, encoded_inputs), model_outputs['loss']
