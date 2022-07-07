# Knowledge Embedding
# Knowledge Selection
from torch import nn


class KnowledgeGrounder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, rationales):
        knowledge_snippets = rationales
        return knowledge_snippets, 0
