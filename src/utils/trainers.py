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

from src.utils.settings import settings


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
                 num_epochs: int,
                 dataset_name: str = None,
                 num_prints: int = 10,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.num_prints = num_prints
        self.dataset_name = dataset_name
        self.results = []

    def save_results(self):
        # Create dictionary with all the parameters
        folder_name_dict = {
            'dataset': self.dataset_name,
            'model': self.model.name if hasattr(self.model, 'name') else self.model.__class__.__name__,
            'num_epochs': self.num_epochs,
        }

        params_dict = {
            'dataset': self.dataset_name,
            'model': str(self.model),
            'optimizer': str(self.optimizer),
            'num_epochs': self.num_epochs,
            'device': settings.device.type,
        }

        # Create a timestamped and args-explicit named for the results folder name
        date = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
        folder_name = '_'.join([date] + [f'{k}={v}' for k, v in folder_name_dict.items()])
        results_path = osp.join(settings.results_dir, 'trainers', folder_name)

        # Create results folder
        os.makedirs(results_path)

        # Plot results
        df = pd.DataFrame(self.results)  # create dataframe
        df.index += 1  # shift index by 1, because epochs start at 1
        for i, col in enumerate(df.columns):
            df[col].plot(fig=plt.figure(i))
            col_name = capitalize(col)
            plt.title(col_name)
            plt.xlabel('Epoch')
            # plt.ylabel(col_name)

            plt.savefig(osp.join(results_path, f'{col}.png'))
            plt.close()

        with open(osp.join(results_path, 'params.json'), 'w') as f:
            json.dump(params_dict, f)

        with open(osp.join(results_path, 'results.json'), 'w') as f:
            json.dump(self.results, f)

        torch.save(self.model.state_dict(), osp.join(results_path, 'model.pt'))

        # Print path to the results directory
        print(f'Results saved to {results_path}')

    def run(self, return_best_epoch_only=True, val_metric='acc'):
        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch}')
            # Train, eval & test
            train_results = self.train()
            eval_results = self.eval()
            test_results = self.test()

            # Save epoch results
            epoch_results = {**train_results, **eval_results, **test_results}

            # Clean epoch results
            epoch_results = {k: v for k, v in epoch_results.items() if v is not None}

            # print epoch and results
            if epoch % (self.num_epochs // min(self.num_epochs, self.num_prints)) == 0:
                self.print_epoch_with_results(epoch, epoch_results)

            # Save epoch results to list
            self.results.append(epoch_results)

        # Print best epoch and results
        best_epoch, best_results = self.get_best_epoch_with_results(val_metric)
        print(f'*** BEST ***')
        self.print_epoch_with_results(best_epoch, best_results)

        # Save results to file
        self.save_results()

        # Return best epoch results i.e. the one w/ the highest validation metric value
        if return_best_epoch_only:
            return best_results
        else:
            # Return best & all epoch results
            return best_results, self.results

    @staticmethod
    def print_epoch_with_results(epoch, epoch_results):
        epoch_results_str = ', '.join([f'{capitalize(k)}: {v:.4f}' for k, v in epoch_results.items()])
        print(f'Epoch: {epoch:02d}, {epoch_results_str}')

    def get_best_epoch_with_results(self, val_metric):
        best_epoch_results = max(self.results, key=lambda x: x.get(f'val_{val_metric}', 0))
        return self.results.index(best_epoch_results) + 1, best_epoch_results