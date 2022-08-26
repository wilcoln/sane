import os.path as osp

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.datasets.nl import get_loader
from src.models.sane import SANE
from src.settings import settings


def compute_knowledge_relevance_index(model, inputs, results_path):
    # run model
    pred, nle, att_knwl, knwl = model(inputs)
    pkr = pred.knowledge_relevance
    ekr = nle.knowledge_relevance

    pkr = pkr.cpu().detach().numpy()

    ekr_factor = 1/ekr.shape[1]
    ekr = ekr.flatten().cpu().detach().numpy()

    # Plot histogram of pkr and ekr
    if settings.use_science:
        plt.style.use('science')

    plt.hist(pkr, bins=np.arange(0, 1.1, 0.05), alpha=0.5, label='Pred KRI', edgecolor='black', linewidth=1.)
    plt.hist(ekr, bins=np.arange(0, 1.1, 0.05), alpha=0.5, weights=ekr_factor*np.ones_like(ekr), label='NLE KRI',
             edgecolor='black', linewidth=1.)
    plt.legend(loc='upper right')
    plt.tight_layout()
    # Save the plot in the results directory
    plt.savefig(osp.join(results_path, 'knowledge_relevance_index.pdf'))
    plt.savefig(osp.join(results_path, 'knowledge_relevance_index.png'))

    # Show the plot
    plt.show()


if __name__ == '__main__':
    # Get test dataloader
    inputs = next(iter(get_loader('val')))

    # Load model
    model = SANE().to(settings.device)
    results_path = settings.input_dir
    model.load_state_dict(torch.load(osp.join(results_path, 'model.pt')))
    model.eval()

    compute_knowledge_relevance_index(model, inputs, results_path)
