import torch
from torch import nn
from utils.settings import settings
loss_fn = nn.CrossEntropyLoss()
from utils.embeddings import bert
from utils.nn import MLP


class Predictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mlp = MLP(in_channels=384*2, out_channels=3, hidden_channels=settings.hidden_dim)

    def forward(self, inputs, nles):

        # Encode premise and hypothesis
        premises_embeddings = bert(inputs['Sentence1'], verbose=False)
        hypothesis_embeddings = bert(inputs['Sentence2'], verbose=False)

        outputs = self.mlp(torch.cat([premises_embeddings, hypothesis_embeddings], dim=1))

        loss = loss_fn(outputs, inputs['gold_label'])

        return outputs, loss
