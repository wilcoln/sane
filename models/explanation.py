from torch import nn
from transformers import BartTokenizer

from utils.embeddings import transformer_mean_pooling
from utils.settings import settings
from utils.transformers import BartForExplanationGeneration, BartForExplanationGenerationWK


class Explainer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = BartForExplanationGeneration.from_pretrained("facebook/bart-base")
        self.model.set_lm_head(knowledge_dim=2 * settings.hidden_dim)
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    def forward(self, inputs):
        encoded_sentences = self.tokenizer(inputs['Sentences'], max_length=512, truncation=True, padding=True,
                                           return_tensors="pt")
        encoded_explanations = self.tokenizer(inputs['Explanation_1'], max_length=512, truncation=True,
                                              padding=True, return_tensors="pt")

        # send tensors to gpu
        encoded_sentences = {k: v.to(settings.device) for k, v in encoded_sentences.items()}
        encoded_explanations = {k: v.to(settings.device) for k, v in encoded_explanations.items()}
        encoded_knowledge = {'knowledge_embedding': inputs['Knowledge_embedding']}
        out_tokens, model_outputs = self.model(**encoded_sentences, **encoded_knowledge,
                                               labels=encoded_explanations['input_ids'])
        return out_tokens, transformer_mean_pooling(model_outputs, encoded_sentences), model_outputs['loss']


class ExplainerWithoutKnowledge(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = BartForExplanationGenerationWK.from_pretrained("facebook/bart-base")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    def forward(self, inputs):
        encoded_inputs = self.tokenizer(inputs['Sentences'], max_length=512, truncation=True, padding=True,
                                        return_tensors="pt")
        encoded_labels = self.tokenizer(inputs['Explanation_1'], max_length=512, truncation=True,
                                        padding=True, return_tensors="pt")

        # send tensors to gpu
        encoded_inputs = {k: v.to(settings.device) for k, v in encoded_inputs.items()}
        encoded_labels = {k: v.to(settings.device) for k, v in encoded_labels.items()}
        out_tokens, model_outputs = self.model(**encoded_inputs, labels=encoded_labels['input_ids'])

        return out_tokens, transformer_mean_pooling(model_outputs, encoded_inputs), model_outputs['loss']
