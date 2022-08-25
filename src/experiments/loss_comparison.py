import json
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from src.settings import settings
from src.utils.format import capitalize


def plot_loss_comparison(results):
    plt.style.use('science')

    keys = ['acc', 'nle_loss']
    fig, axes = plt.subplots(1, len(keys))
    fig.set_size_inches(8, 4)
    for key, ax in zip(keys, axes):
        loss = []
        loss_nk = []

        for i, entry in enumerate(results):
            if i == settings.num_epochs:
                break
            loss.append(entry[f'val_{key}'])
            loss_nk.append(entry[f'val_{key}_nk'])

        epochs = np.arange(1, len(loss) + 1)

        ax.plot(epochs, loss, label='w/ Knowledge', marker='o')
        ax.plot(epochs, loss_nk, label='w/o Knowledge', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(capitalize(key))
        ax.grid()

    lines, labels = fig.axes[-1].get_legend_handles_labels()

    fig.legend(lines, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()

    # Save the plot in the results directory
    plt.savefig(osp.join(results_path, 'loss_comparison.png'))

    # Show the plot
    plt.show()


if __name__ == '__main__':
    results_path = settings.input_dir
    # Load results json
    with open(osp.join(results_path, 'results.json'), 'r') as f:
        results = json.load(f)

    plot_loss_comparison(results)
