import os.path as osp

import pandas as pd
import torch

from src.conceptnet import conceptnet
from src.datasets.nl import get_loader
from src.models.sane import SANE
from src.settings import settings
from src.utils.embeddings import tokenizer


def compute_knowledge_attention_map(inputs, model):
    # run model
    pred, nle, att_knwl, knwl = model(inputs)
    triples = conceptnet.ids2triples(knwl.id.tolist())
    sentences = tokenizer.batch_decode(inputs['Sentences']['input_ids'].to(settings.device), skip_special_tokens=True)
    triples = [' '.join(t) for t in triples]

    # Get top k triples and their attention weightstopk_triples = []
    # topk = 20
    # topk_attentions, indices = torch.topk(att_knwl.attentions, topk, sorted=False)
    # topk_attentions = F.softmax(topk_attentions)  # re-normalize
    # topk_triple_ids = knwl.id[indices]
    # top_k_raw_triples = [conceptnet.ids2triples(ids.tolist()) for ids in topk_triple_ids]
    # topk_triples.extend((top_k_raw_triples, topk_attentions.tolist()))

    # Save attention maps
    for i in range(settings.num_attn_heads):
        np_attention = att_knwl.weights.cpu().detach().numpy()
        df = pd.DataFrame(np_attention, index=sentences, columns=triples)
        csv_path = osp.join(results_path, f'knowledge_attention_map_{settings.batch_size}x{settings.max_concepts_per_sent}_{i + 1}.csv')
        df.to_csv(csv_path)


if __name__ == '__main__':
    # Get test dataloader
    inputs = next(iter(get_loader('val')))

    # Load model
    model = SANE().to(settings.device)
    results_path = settings.input_dir
    model.load_state_dict(torch.load(osp.join(results_path, 'model.pt')))
    model.eval()

    compute_knowledge_attention_map(inputs, model)
