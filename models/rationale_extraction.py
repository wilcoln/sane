from torch import nn


class RationaleExtractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs):
        return inputs, 0
