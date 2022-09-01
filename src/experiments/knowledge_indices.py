import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.datasets.nl import get_loader
from src.models.sane import SANE
from src.settings import settings


def compute_knowledge_indices(model, results_path, dataloader):
    pki_dict = {'relevance': [], 'contribution': []}
    eki_dict = {'relevance': [], 'contribution': []}

    pdf_path, png_path = {}, {}
    for index in pki_dict.keys():
        pdf_path[index] = osp.join(results_path, f'knowledge_{index}_index.pdf')
        png_path[index] = osp.join(results_path, f'knowledge_{index}_index.png')

    # run model
    for i, inputs in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        pred, nle, _, _ = model(inputs)
        for index in {'relevance', 'contribution'}:
            pki_dict[index].append(getattr(pred, f'knowledge_{index}'))
            eki_dict[index].append(getattr(nle, f'knowledge_{index}'))

    pki_dict = {k: torch.cat(v, dim=0) for k, v in pki_dict.items()}
    eki_dict = {k: torch.cat(v, dim=0) for k, v in eki_dict.items()}

    for index in {'relevance', 'contribution'}:
        pki = pki_dict[index].cpu().detach().numpy()
        eki_factor = 1 / eki_dict[index].shape[1]
        eki = eki_dict[index].flatten().cpu().detach().numpy()

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
        plt.savefig(pdf_path[index])
        plt.savefig(png_path[index])

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
