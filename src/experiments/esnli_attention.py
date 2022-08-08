import os.path as osp

import pandas as pd
import torch

from src.conceptnet import conceptnet
from src.datasets.esnli import get_loader
from src.models.kax import KAX
from src.settings import settings
from src.utils.embeddings import tokenizer

# Get test dataloader
inputs = next(iter(get_loader('test', num_chunks=1)))

# Load model
model = KAX().to(settings.device)
results_path = 'results/trainers/2022-08-06_14-15-44_659230_dataset=ESNLI_model=KAX_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=32_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4'
model.load_state_dict(torch.load(osp.join(results_path, 'model.pt')))
model.eval()

# run model
knwl, att_knwl, _, pred = model(inputs)
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
    np_attention = att_knwl.attentions.cpu().detach().numpy()
    df = pd.DataFrame(np_attention, index=sentences, columns=triples)
    csv_path = osp.join(results_path, f'attention{i + 1}.csv')
    df.to_csv(csv_path)
