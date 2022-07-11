from torch import nn

from utils.settings import settings
from transformers import BartTokenizer, BartForConditionalGeneration


class Explainer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    def forward(self, inputs):
        inputs['Sentences'] = [f'{sent1} -> {sent2}' for sent1, sent2 in zip(inputs['Sentence1'], inputs['Sentence2'])]
        bart_inputs = self.tokenizer(inputs['Sentences'], max_length=1024, truncation=True, padding=True,
                                     return_tensors="pt")
        labels = self.tokenizer(inputs['Explanation_1'], max_length=1024, truncation=True,
                                padding=True, return_tensors="pt")

        # send tensors to gpu
        bart_inputs = {k: v.to(settings.device) for k, v in bart_inputs.items()}
        labels = {k: v.to(settings.device) for k, v in labels.items()}

        outputs = self.model(**bart_inputs, labels=labels['input_ids'])

        return outputs['encoder_last_hidden_state'], outputs['loss']
