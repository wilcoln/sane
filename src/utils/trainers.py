import json
import os
import os.path as osp
import time
from abc import ABC
from datetime import datetime as dt

import pandas as pd
import torch
from icecream import ic
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from src.settings import settings, exp_settings
from src.utils.format import fmt_stats_dict, capitalize
from src.utils.regret import regret


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
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.results_path = None
        self.folder_name_dict = None
        self.best_epoch = 0
        self.best_model_state_dict = None
        self.model = model
        self.optimizer = optimizer
        self.dataset_name = dataset_name
        self.results = []

    def save_params_and_prepare_to_save_results(self):
        # Create dictionary with all the parameters
        params_dict = {
            'dataset': self.dataset_name,
            'model': self.model.__class__.__name__,
        }

        exp_settings_names = [s[0] for s in exp_settings]
        params_dict.update({k: v for k, v in vars(settings).items() if k in exp_settings_names and v})

        # Create a timestamped and args-explicit named for the results' folder name
        date = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
        folder_name = '_'.join([date] + [f'{k}={v}' for k, v in params_dict.items() if v is not None])
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
        if not settings.no_save:
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
            if not settings.no_save:
                self.save_results()

            # print epoch and results
            self.print_epoch(epoch)

        # Print best epoch and results
        print(f'*** BEST ***')
        self.print_epoch(self.best_epoch)

        # Print path to the results directory
        print(f'Results saved to {self.results_path}')

    def print_epoch(self, epoch):
        stats_dict = self.results[epoch - 1]
        epoch_results_str = fmt_stats_dict(stats_dict)
        print(f'Epoch: {epoch:02d}, {epoch_results_str}')


class SANETrainer(TorchModuleBaseTrainer):
    def __init__(self, model, optimizer, dataset_name, train_loader, val_loader, test_loader=None, expert=None):
        super().__init__(model, optimizer, dataset_name)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.expert = expert

    def evaluate(self, dataloader, split):
        """Train, Evaluate, and Test the Model"""

        description = {'train': 'Training', 'val': 'Validation', 'test': 'Testing'}
        if dataloader is None:
            print(f'Skipping {description[split]}')
            return {}

        assert split in {'train', 'val', 'test'}, "split must be either 'train', 'val' or 'test'"

        train = split == 'train' and (not settings.frozen)

        self.model.train() if train else self.model.eval()

        # Set split loss value
        split_loss = 0.0

        # Reset values for accuracy computation
        correct = 0
        total = 0

        # Iterate over the DataLoader for training data
        pbar = tqdm(enumerate(dataloader, 0), total=len(dataloader))
        pbar.set_description(description[split])
        start = time.time()
        for i, inputs in pbar:
            ic(inputs['Sentences_embedding'].requires_grad)
            if train:
                # zero the parameter gradients
                self.optimizer.zero_grad()

            # forward pass & compute loss
            pred, nle = self.model(inputs)[:2]

            # Compute loss
            loss = settings.alpha * nle.loss + (1 - settings.alpha) * pred.loss

            if train:
                if self.expert is not None:
                    # forward + backward on expert
                    expert_pred, expert_nle = self.expert(inputs)[:2]
                    # Compute regret
                    pred_regret, nle_regret = regret(pred.loss, expert_pred.loss), regret(nle.loss, expert_nle.loss)
                    # Add regret to loss
                    loss += settings.alpha * nle_regret + settings.alpha * pred_regret

                # backward pass + optimization step
                loss.backward()
                self.optimizer.step()

            # Update Loss
            split_loss += loss.item()

            # Update Accuracy
            predicted = pred.logits.argmax(1)
            total += inputs['gold_label'].size(0)
            correct += predicted.eq(inputs['gold_label'].to(settings.device)).sum().item()

        split_time = time.time() - start
        split_loss /= len(dataloader)
        split_acc = 100. * correct / total

        return {
            f'{split}_acc': split_acc,
            f'{split}_loss': split_loss,
            f'{split}_time': split_time,
        }

    def train(self) -> dict:
        return self.evaluate(self.train_loader, 'train')

    @torch.no_grad()
    def eval(self) -> dict:
        return self.evaluate(self.val_loader, 'val')

    @torch.no_grad()
    def test(self) -> dict:
        return self.evaluate(self.test_loader, 'test')
