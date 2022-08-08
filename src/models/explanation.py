from torch import nn

from src.settings import settings
from src.utils.transformers import BartForExplanationGeneration


class Explainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BartForExplanationGeneration.from_pretrained("facebook/bart-base")
        self.model.set_fusion_head(knowledge_dim=settings.sent_dim)

    def forward(self, inputs, knowledge=None):
        # send tensors to gpu
        encoded_inputs = {k: v.to(settings.device) for k, v in inputs['Sentences'].items()}
        encoded_labels = {k: v.to(settings.device) for k, v in inputs['Explanation_1'].items()}
        encoded_knowledge = {'knowledge_embedding': knowledge}
        return self.model(**encoded_inputs, **encoded_knowledge, labels=encoded_labels['input_ids'])
