from torch import nn

from utils.embeddings import transformer_mean_pooling
from utils.settings import settings
from transformers import BartTokenizer
from utils.transformers import BartForExplanationGeneration


class Explainer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = BartForExplanationGeneration.from_pretrained("facebook/bart-large")
        self.model.knowledge_d = settings.hidden_dim
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    def forward(self, inputs):
        inputs['Sentences'] = [f'{sent1} -> {sent2}' for sent1, sent2 in zip(inputs['Sentence1'], inputs['Sentence2'])]
        encoded_inputs = self.tokenizer(inputs['Sentences'], max_length=1024, truncation=True, padding=True,
                                     return_tensors="pt")
        encoded_labels = self.tokenizer(inputs['Explanation_1'], max_length=1024, truncation=True,
                                padding=True, return_tensors="pt")

        # send tensors to gpu
        encoded_inputs = {k: v.to(settings.device) for k, v in encoded_inputs.items()}
        encoded_labels = {k: v.to(settings.device) for k, v in encoded_labels.items()}
        encoded_knowledge = {'knowledge_embedding': inputs['Knowledge_embedding']}
        model_outputs = self.model(**encoded_inputs, **encoded_knowledge, labels=encoded_labels['input_ids'])

        return transformer_mean_pooling(model_outputs, encoded_inputs), model_outputs['loss']
