import os.path as osp

import pandas as pd
import torch
import torch.nn.functional as F
from icecream import ic
from tqdm import tqdm

from src.conceptnet import conceptnet
from src.datasets.esnli import get_loader
from src.models.kax import KAX
from src.settings import settings
import matplotlib.pyplot as plt
import numpy as np

# Get test dataloader
dataloader = get_loader('test')

# Load model
model = KAX().to(settings.device)
results_path = 'results/trainers/2022-08-06_14-15-44_659230_dataset=ESNLI_model=KAX_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=32_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4'
model.load_state_dict(torch.load(osp.join(results_path, 'model.pt')))
model.eval()

attentions = []
topk_triples = []
topk = 20

# Run model on test set
for i, inputs in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
    # run model
    knwl, fused_knwl, _, pred = model(inputs)

    # Get top k triples and their attention weights
    topk_attentions, indices = torch.topk(fused_knwl.attentions, topk, sorted=False)
    topk_attentions = F.softmax(topk_attentions)  # re-normalize
    topk_triple_ids = knwl.id[indices]
    top_k_raw_triples = [conceptnet.ids2triples(ids.tolist()) for ids in topk_triple_ids]
    topk_triples.extend((top_k_raw_triples, topk_attentions.tolist()))
    attentions.append(fused_knwl.attentions)
    break


# Save attention maps
img_path = osp.join(results_path, 'attention.png')
plt.imsave(img_path, attentions[0].cpu().detach().numpy(), cmap='hot')
# ic(topk_triples[:10])

