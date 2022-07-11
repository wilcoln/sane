import settings
from torch import nn

from utils.settings import settings
from transformers import BartTokenizer, BartForConditionalGeneration


class Explainer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    def forward(self, inputs):
        inputs['Sentences'] = [f'{sent1} -> {sent2}' for sent1, sent2 in zip(inputs['Sentence1'], inputs['Sentence2'])]
        bart_inputs = self.tokenizer(inputs['Sentences'], max_length=1024, truncation=True, padding=True,
                                     return_tensors="pt")
        labels = self.tokenizer(inputs['Explanation_1'], max_length=1024, truncation=True,
                                padding=True, return_tensors="pt")

        # Generate Summary
        # summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
        # ic(self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

        # send tensors to gpu
        bart_inputs = {k: v.to(settings.device) for k, v in bart_inputs.items()}
        labels = {k: v.to(settings.device) for k, v in labels.items()}

        outputs = self.model(**bart_inputs, labels=labels['input_ids'])

        return outputs['encoder_last_hidden_state'], outputs['loss']
