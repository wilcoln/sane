import json
import os
import os.path as osp
from abc import ABC
from datetime import datetime as dt

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Optimizer

from src.settings import settings


def capitalize(underscore_string):
    return ' '.join(w.capitalize() for w in underscore_string.split('_'))


class BaseTrainer:
    def __init__(self, dataset_name: str = None):
        self.dataset_name = dataset_name

    def train(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    def eval(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    def test(self, *args, **kwargs) -> dict:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> dict:
        raise NotImplementedError


class TorchModuleBaseTrainer(BaseTrainer, ABC):
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 dataset_name: str = None,
                 num_prints: int = 10,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.results_path = None
        self.folder_name_dict = None
        self.best_epoch = 0
        self.best_model_state_dict = None
        self.model = model
        self.optimizer = optimizer
        self.num_prints = num_prints
        self.dataset_name = dataset_name
        self.results = []

    def save_params_and_prepare_to_save_results(self):
        # Create dictionary with all the parameters
        params_dict = {
            'dataset': self.dataset_name,
            'model': self.model.__class__.__name__,
        }
        params_dict.update({k: v for k, v in vars(settings).items() if k in settings.exp_settings[0] and v})

        # Create a timestamped and args-explicit named for the results' folder name
        date = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
        folder_name = '_'.join([date] + [f'{k}={v}' for k, v in self.folder_name_dict.items() if v is not None])
        self.results_path = osp.join(settings.results_dir, 'trainers', folder_name)

        # Create results folder
        os.makedirs(self.results_path)

        with open(osp.join(self.results_path, 'params.json'), 'w') as f:
            json.dump(params_dict, f)

    def save_results(self):
        # Plot results
        df = pd.DataFrame(self.results)  # create dataframe
        df.index += 1  # shift index by 1, because epochs start at 1
        for i, col in enumerate(df.columns):
            df[col].plot(fig=plt.figure(i))
            col_name = capitalize(col)
            plt.title(col_name)
            plt.xlabel('Epoch')
            # plt.ylabel(col_name)

            plt.savefig(osp.join(self.results_path, f'{col}.png'))
            plt.close()

        with open(osp.join(self.results_path, 'results.json'), 'w') as f:
            json.dump(self.results, f)

        torch.save(self.best_model_state_dict, osp.join(self.results_path, 'model.pt'))

    def run(self, val_metric='acc'):
        self.save_params_and_prepare_to_save_results()

        for epoch in range(1, settings.num_epochs + 1):
            print(f'Epoch {epoch}')
            # Train, eval & test
            train_results = self.train()
            eval_results = self.eval()
            test_results = self.test()

            # Save epoch results
            epoch_results = {**train_results, **eval_results, **test_results}

            # Save best model epoch
            val_metric_key = f'val_{val_metric}'
            if not self.best_epoch or epoch_results[val_metric_key] > self.results[self.best_epoch - 1][val_metric_key]:
                self.best_epoch = epoch
                self.best_model_state_dict = self.model.state_dict()

            # Clean epoch results
            epoch_results = {k: v for k, v in epoch_results.items() if v is not None}

            # Save epoch results to list
            self.results.append(epoch_results)

            # Save results to file
            self.save_results()

            # print epoch and results
            if epoch % (settings.num_epochs // min(settings.num_epochs, self.num_prints)) == 0:
                self.print_epoch(epoch)

        # Print best epoch and results
        print(f'*** BEST ***')
        self.print_epoch(self.best_epoch)

        # Print path to the results directory
        print(f'Results saved to {self.results_path}')

    def print_epoch(self, epoch):
        epoch_results_str = ', '.join([f'{capitalize(k)}: {v:.4f}' for k, v in self.results[epoch - 1].items() if not
        k.endswith(
            '_time')])
        print(f'Epoch: {epoch:02d}, {epoch_results_str}')


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'
