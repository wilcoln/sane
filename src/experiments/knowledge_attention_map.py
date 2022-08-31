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


def compute_knowledge_attention_map(model, results_path):
    # Check if results already exist
    csv_path = osp.join(results_path, f'knowledge_attention_map.csv')
    if osp.exists(csv_path):
        return

    inputs = next(iter(get_loader('test', batch_size=5)))
    tmp = settings.max_concepts_per_sent
    settings.max_concepts_per_sent = 5  # Set max concepts per sentence to 5

    # run model
    pred, nle, att_knwl, knwl = model(inputs)
    triples = conceptnet.ids2triples(knwl.id.tolist())
    sentences = frozen_bart_tokenizer.batch_decode(inputs['Sentences']['input_ids'].to(settings.device),
                                                   skip_special_tokens=True)
    triples = [' '.join(t) for t in triples]

    # Save attention maps
    np_attention = att_knwl.weights.cpu().detach().numpy()
    df = pd.DataFrame(np_attention, index=sentences, columns=triples)
    df.to_csv(csv_path)

    settings.max_concepts_per_sentence = tmp  # Reset max concepts per sentence


def plot_knowledge_attention_map(results_path):
    # Check if results already exist
    pdf_path = osp.join(results_path, 'knowledge_attention_map.pdf')
    png_path = osp.join(results_path, 'knowledge_attention_map.png')
    if osp.exists(pdf_path) and osp.exists(png_path):
        return

    # Load attention maps
    csv_path = osp.join(results_path, f'knowledge_attention_map.csv')
    df = pd.read_csv(csv_path, index_col=[0])

    column_mapping = {t: f'T{i}' for i, t in enumerate(df.columns)}
    index_mapping = {s: f'S{i}' for i, s in enumerate(df.index)}
    df.rename(columns=column_mapping, inplace=True)
    df.index = list(index_mapping.values())
    # Plot attention maps
    fig, ax = plt.subplots(figsize=(20, 4))
    sns.set(rc={'text.usetex': True})
    sns.heatmap(df, ax=ax, cmap='viridis')
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.show()


if __name__ == '__main__':
    # Load model
    model = SANE().to(settings.device)
    results_path = settings.input_dir
    model.load_state_dict(torch.load(osp.join(results_path, 'model.pt')))
    model.eval()
    compute_knowledge_attention_map(model, results_path)
    plot_knowledge_attention_map(results_path)
