import os.path as osp

import pandas as pd
import torch
from matplotlib import pyplot as plt

from src.conceptnet import conceptnet
from src.datasets.nl import get_loader
from src.models.sane import SANE
import seaborn as sns
from src.settings import settings
from src.utils.embeddings import frozen_bart_tokenizer


def compute_knowledge_attention_map(inputs, model):
    # run model
    pred, nle, att_knwl, knwl = model(inputs)
    triples = conceptnet.ids2triples(knwl.id.tolist())
    sentences = frozen_bart_tokenizer.batch_decode(inputs['Sentences']['input_ids'].to(settings.device),
                                                   skip_special_tokens=True)
    triples = [' '.join(t) for t in triples]

    # Save attention maps
    np_attention = att_knwl.weights.cpu().detach().numpy()
    df = pd.DataFrame(np_attention, index=sentences, columns=triples)
    suffix = f'{settings.batch_size}x{settings.max_concepts_per_sent}'
    csv_name = f'knowledge_attention_map_{suffix}.csv'
    csv_path = osp.join(results_path, csv_name)
    df.to_csv(csv_path)
    return csv_path


def plot_knowledge_attention_map(results_path, csv_name):
    # Load attention maps
    df = pd.read_csv(osp.join(results_path, csv_name), index_col=[0])

    column_mapping = {t: f'T{i}' for i, t in enumerate(df.columns)}
    index_mapping = {s: f'S{i}' for i, s in enumerate(df.index)}
    df.rename(columns=column_mapping, inplace=True)
    df.index = list(index_mapping.values())
    # Plot attention maps
    fig, ax = plt.subplots(figsize=(20, 4))
    sns.set(rc={'text.usetex': True})
    sns.heatmap(df, ax=ax, cmap='viridis')
    plt.tight_layout()
    plt.savefig(osp.join(results_path, 'knowledge_attention_map.pdf'))
    plt.savefig(osp.join(results_path, 'knowledge_attention_map.png'))
    plt.show()


if __name__ == '__main__':
    # Get test dataloader
    inputs = next(iter(get_loader('val')))

    # Load model
    model = SANE().to(settings.device)
    results_path = settings.input_dir
    model.load_state_dict(torch.load(osp.join(results_path, 'model.pt')))
    model.eval()
    csv_name = compute_knowledge_attention_map(inputs, model)
    plot_knowledge_attention_map(results_path, csv_name)
