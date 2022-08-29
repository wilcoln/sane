import json
import os
import os.path as osp
from datetime import datetime as dt

import numpy as np
import torch
from icecream import ic
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.datasets import fetch_california_housing
from torch import nn
from torch.utils.data import random_split
from tqdm import tqdm

from src.settings import settings
from src.utils.format import capitalize
from src.utils.regret import regret
from src.utils.trainers import TorchModuleBaseTrainer


class f(nn.Module):
    def __init__(self, h_dim, y_dim, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels if hidden_channels else min(h_dim, y_dim)
        self.model = nn.Sequential(
            nn.Linear(h_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, y_dim),
        )

    def forward(self, x):
        return self.model(x)


class h(nn.Module):
    def __init__(self, x_dim, h_dim, hidden_channels=None):
        hidden_channels = hidden_channels if hidden_channels else min(x_dim, h_dim)
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, h_dim),
        )

    def forward(self, x):
        return self.model(x)


class g(nn.Module):
    def __init__(self, x_dim, k_dim, h_dim, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels if hidden_channels else min(x_dim, k_dim, h_dim)
        self.g2 = nn.Sequential(
            nn.Linear(x_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, h_dim),
        )
        self.g1 = nn.Sequential(
            nn.Linear(x_dim + k_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, k):
        return self.g1(torch.hstack((x, k))), self.g2(x)


class g0(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, _):
        return torch.zeros(x.shape[0], 1).to(settings.device), torch.zeros_like(x).to(settings.device)


class SimpleSANE(nn.Module):
    def __init__(self, x_dim, k_dim, y_dim, h_dim):
        super().__init__()
        self.f = f(h_dim, y_dim)
        self.g = g(x_dim, k_dim, h_dim)
        self.h = h(x_dim, h_dim)

    def forward(self, x, k):
        # Compute yhat
        x_tilde = self.h(x)
        r, k_tilde = self.g(x, k)
        rk_tilde = r * k_tilde
        x_tilde_plus = x_tilde + rk_tilde
        yhat = self.f(x_tilde_plus)

        # Compute kri
        ck = torch.norm(rk_tilde, dim=1)**2
        cx = torch.norm(x_tilde, dim=1)**2
        c = ck / (ck + cx)

        # Return yhat and kri
        return yhat, r, c


class SimpleSANENoKnowledge(SimpleSANE):
    def __init__(self, x_dim, k_dim, y_dim, h_dim):
        super().__init__(x_dim, k_dim, y_dim, h_dim)
        self.g = g0(x_dim, k_dim, h_dim)


# Train the model and print loss and accuracy on the train and validation sets
class Trainer(TorchModuleBaseTrainer):
    def __init__(self, model, model_nk, optimizer, optimizer_nk, train_loader, val_loader, loss_fn, loss_fn_nk,
                 params):
        super().__init__(model, optimizer, train_loader, val_loader, params)
        self.loss_fn = loss_fn
        self.loss_fn_nk = loss_fn_nk
        self.model_nk = model_nk
        self.optimizer_nk = optimizer_nk

    def evaluate(self, dataloader, split):
        """Train, Evaluate, and Test the Model"""

        if dataloader is None:
            print(f'Skipping {self.split_descriptions[split]}')
            return {}

        train = split == 'train' and (not settings.frozen)

        self.model.train() if train else self.model.eval()

        # Set split loss values
        split_loss = 0.0
        split_loss_nk = 0.0
        # Set split knowledge relevance & contribution indices
        split_kri = 0.0
        split_kci = 0.0

        # Reset values for accuracy computation
        correct = 0
        correct_nk = 0
        total = 0

        # Iterate over the DataLoader for training data
        pbar = tqdm(enumerate(dataloader, 0), total=len(dataloader))
        pbar.set_description(self.split_descriptions[split])
        for i, (x, k, y) in pbar:
            x, k, y = x.to(settings.device), k.to(settings.device), y.to(settings.device)

            #####################################
            # (1) Compute loss without knowledge
            #####################################
            # zero the parameter gradients
            self.optimizer_nk.zero_grad()

            # forward pass & compute loss without knowledge
            yhat, _, _ = self.model_nk(x, k)
            # Compute loss knowledge without knowledge
            loss_nk = loss_fn(yhat, y)

            if train:
                # backward pass + optimization step
                loss_nk.mean().backward()
                self.optimizer_nk.step()

            # Update Split Loss no knowledge
            split_loss_nk += loss_nk.mean().item()
            # Update Accuracy
            yhat = yhat.argmax(1)
            correct_nk += yhat.eq(y).sum().item()

            # clean vars and gradients after step
            self.optimizer_nk.zero_grad()
            del yhat

            #################################################
            # (2) Compute regret-augmented loss with knowledge
            ##################################################
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward pass
            yhat, kri, kci = self.model(x, k)
            # Compute loss with knowledge
            loss = loss_fn(yhat, y)

            # Compute regret loss
            regret_loss = regret(loss, loss_nk.detach(), reduce=False)

            # Compute regret-augmented loss with knowledge
            augmented_loss = (1 - settings.beta) * loss + settings.beta * regret_loss

            if train:
                # backward pass + optimization step
                augmented_loss.mean().backward()
                self.optimizer.step()

            # Update Split Loss
            split_loss += loss.mean().item()
            # Update Split Knowledge relevance
            split_kri += kri.mean().item()
            # Update Split Knowledge contribution
            split_kci += kci.mean().item()
            # Update Accuracy
            yhat = yhat.argmax(1)
            correct += yhat.eq(y).sum().item()

            # clean vars and gradients after step
            self.optimizer.zero_grad()
            del yhat, loss, loss_nk

            # Update total
            total += len(y)

        split_loss /= len(dataloader)
        split_loss_nk /= len(dataloader)
        split_kri /= len(dataloader)
        split_kci /= len(dataloader)
        split_acc = 100. * correct / total
        split_acc_nk = 100. * correct_nk / total

        return {
            f'{split}_acc': split_acc,
            f'{split}_acc_nk': split_acc_nk,
            f'{split}_loss': split_loss,
            f'{split}_loss_nk': split_loss_nk,
            f'{split}_kri': split_kri,  # knowledge relevance index
            f'{split}_kci': split_kci,  # knowledge consistency index
        }


if __name__ == '__main__':
    torch.manual_seed(0)

    settings.no_save = True

    params = {
        'num_epochs': settings.num_epochs,
        'num_runs': settings.num_runs,
        'train_size': 0.8,
        'batch_size': 1024,
        'lr': 5e-3,
        'input': 'pca',
    }

    date = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
    folder_name = '_'.join([date] + [f'{k}={v}' for k, v in params.items() if v is not None])
    results_path = osp.join(settings.results_dir, 'kannada_mnist', folder_name)[:200]

    # Create results folder
    os.makedirs(osp.join(results_path, 'pdf'))
    with open(osp.join(results_path, 'params.json'), 'w') as file:
        json.dump(params, file)

    _slice = {
        'pca': slice(0, 2),
        'tsne': slice(2, 4),
        'umap': slice(4, 6),
    }
    results = {}
    keys = None
    knowledge_list = ['gaussian', 'confusing', 'tsne', 'umap']

    # Load the kannada mnist dataset from data_dir
    data_dir = osp.join(settings.data_dir, 'kannada_mnist_2d_pca_tsne_umap')
    x_k = torch.from_numpy(np.load(osp.join(data_dir, 'x.npy'))).float()
    y = torch.from_numpy(np.load(osp.join(data_dir, 'y.npy')))
    # Set input features
    x = x_k[:, _slice[params['input']]]

    for knowledge in tqdm(knowledge_list):
        # Set knowledge features
        if knowledge == 'gaussian':
            k = torch.randn_like(x)
        elif knowledge == 'confusing':
            housing = fetch_california_housing().data[:x.shape[0]]
            k = torch.from_numpy(housing[:x.shape[0]]).float()
        else:
            k = x_k[:, _slice[knowledge]]
        # Create dataset
        dataset = torch.utils.data.TensorDataset(x, k, y)

        # Split the data into train and validation sets
        train_size = int(params['train_size'] * len(dataset))
        train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=params['batch_size'], shuffle=True)

        # Set the dimensions
        x_dim = x.shape[1]
        k_dim = k.shape[1]
        y_dim = len(torch.unique(y))
        h_dim = x_dim

        run_results = []
        for _ in range(params['num_runs']):
            # Create the models
            model = SimpleSANE(x_dim, k_dim, y_dim, h_dim).to(settings.device)
            model_nk = SimpleSANENoKnowledge(x_dim, k_dim, y_dim, h_dim).to(settings.device)

            # Create the optimizers
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])
            optimizer_nk = torch.optim.AdamW(model_nk.parameters(), lr=params['lr'])

            # Create the loss functions
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss_fn_nk = nn.CrossEntropyLoss(reduction='none')

            # Create the trainer
            trainer = Trainer(model, model_nk, optimizer, optimizer_nk, train_loader, val_loader, loss_fn,
                              loss_fn_nk, params)

            # Train the model
            trainer.run(val_metric='acc', less_is_more=False)

            # Add results to list
            run_results.append(trainer.results)

        if keys is None:
            keys = run_results[0][0].keys()

        # Compute mean and std
        means = {
            key: np.array([
                np.mean([result[epoch_idx][key] for result in run_results])
                for epoch_idx in range(params['num_epochs'])
            ]) for key in keys
        }

        stds = {
            key: np.array([
                np.std([result[epoch_idx][key] for result in run_results])
                for epoch_idx in range(params['num_epochs'])
            ]) for key in keys
        }

        # Save results
        results[knowledge] = {'runs': run_results, 'mean': means, 'std': stds}

    # Plot results with shaded region for std
    if settings.use_science:
        plt.style.use('science')
    else:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
        })

    metrics = {key.split('_')[1] for key in keys}
    suffix_style_map = {'': 'solid', '_nk': 'dashed'}
    suffix_legend_map = {'': '$f\circ(h + g_1\cdot g_2)$', '_nk': '$p$'}
    knowledge_color_map = {knowledge: plt.get_cmap('tab10')(i) for i, knowledge in enumerate(knowledge_list)}
    x = range(params['num_epochs'])

    model_legend = [
        Line2D([0], [0], color='black', label=capitalize(suffix_legend_map[suffix]), linestyle=style)
        for suffix, style in suffix_style_map.items()
    ]

    knowledge_legend = [
        Line2D([0], [0], color=color, lw=4, label=capitalize(knowledge))
        for knowledge, color in knowledge_color_map.items()
    ]

    legends = knowledge_legend + model_legend
    for split in {'train', 'val'}:
        for metric in metrics:
            for knowledge in knowledge_list:
                for suffix in suffix_style_map.keys():
                    if metric in {'kri', 'kci'} and suffix == '_nk':
                        continue
                    key = f'{split}_{metric}{suffix}'
                    means = results[knowledge]['mean'][key]
                    stds = results[knowledge]['std'][key]
                    color = knowledge_color_map[knowledge]
                    linestyle = suffix_style_map[suffix]
                    plt.plot(x, means, color=color, label=knowledge, linestyle=linestyle)
                    plt.fill_between(x, means - stds, means + stds, alpha=0.1)
            plt.ylabel(capitalize(metric))
            plt.xlabel('Epoch')
            plt.legend(handles=legends, loc='upper right')
            plt.tight_layout()

            plt.savefig(osp.join(results_path, f'{split}_{metric}.png'))
            plt.savefig(osp.join(results_path, 'pdf', f'{split}_{metric}.pdf'))
            plt.close()

    # Save results
    # with open(osp.join(results_path, 'results.json'), 'w') as file:
    #     json.dump(json.dumps(results), file)

    print(f'Results saved to {results_path}')
