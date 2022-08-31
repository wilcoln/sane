import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datasets.nl import get_loader
from src.models.sane import SANE
from src.settings import settings


def compute_knowledge_indices(model, inputs, results_path):
    # run model
    pred, nle, att_knwl, knwl = model(inputs)
    for index in ['relevance', 'contribution']:
        pki = getattr(pred, f'knowledge_{index}')
        eki = getattr(nle, f'knowledge_{index}')
        pki = pki.cpu().detach().numpy()

        eki_factor = 1 / eki.shape[1]
        eki = eki.flatten().cpu().detach().numpy()

        # Plot histogram of pki and eki
        if settings.use_science:
            plt.style.use('science')

        abbr = f'k{index[0]}i'.upper()
        p_label = f'Pred {abbr}'
        e_label = f'NLE (Token) {abbr}'
        min_ = min(pki.min(), eki.min())
        max_ = max(pki.max(), eki.max())
        bins = np.arange(0, 1.1, 1./60) if index == 'relevance' else np.linspace(min_, max_, 60)
        plt.hist(pki, bins=bins, alpha=0.5, label=p_label, edgecolor='black', linewidth=1.)
        plt.hist(eki, bins=bins, alpha=0.5, weights=eki_factor*np.ones_like(eki), label=e_label,
                 edgecolor='black', linewidth=1.)

        plt.legend(loc='upper right')
        plt.tight_layout()
        # Save the plot in the results directory
        plt.savefig(osp.join(results_path, f'knowledge_{index}_index.pdf'))
        plt.savefig(osp.join(results_path, f'knowledge_{index}_index.png'))

        # Print results dir
        print(f'Results saved to {results_path}')

        # Show the plot
        if settings.show_plots:
            plt.show()
        plt.close()


if __name__ == '__main__':
    # Get test dataloader
    inputs = next(iter(get_loader('val')))
    # Load model
    model = SANE().to(settings.device)
    results_path = settings.input_dir
    model.load_state_dict(torch.load(osp.join(results_path, 'model.pt')))
    model.eval()
    compute_knowledge_indices(model, inputs, results_path)
