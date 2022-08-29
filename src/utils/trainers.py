import json
import os
import os.path as osp
import time
from abc import ABC
from datetime import datetime as dt

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.settings import settings
from src.utils.format import fmt_stats_dict, capitalize
from src.utils.regret import regret


class TorchModuleBaseTrainer(ABC):
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 params: dict,
                 test_loader: DataLoader = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.results_path = None
        self.folder_name_dict = None
        self.best_epoch = 0
        self.best_model_state_dict = None
        self.model = model
        self.optimizer = optimizer
        self.params = params
        self.results = []
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.split_descriptions = {'train': 'Training', 'val': 'Validation', 'test': 'Testing'}

    def save_params_and_prepare_to_save_results(self):
        # Create dictionary with all the parameters
        params_dict = {'model': self.model.__class__.__name__}

        params_dict.update(self.params)

        # Create a timestamped and args-explicit named for the results' folder name
        date = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
        folder_name = '_'.join([date] + [f'{k}={v}' for k, v in params_dict.items() if v is not None])
        self.results_path = osp.join(settings.results_dir, 'trainers', folder_name)[:200]

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

    def run(self, val_metric='loss', less_is_more=True):
        if not settings.no_save:
            self.save_params_and_prepare_to_save_results()

        for epoch in range(1, self.params['num_epochs'] + 1):
            print(f'Epoch {epoch}')
            # Train, eval & test
            train_results = self.train()
            eval_results = self.eval()
            test_results = self.test()

            # Save epoch results
            epoch_results = {**train_results, **eval_results, **test_results}

            # Save best model epoch
            if self.are_best_results(epoch_results, f'val_{val_metric}', less_is_more):
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
        if not settings.no_save:
            print(f'Results saved to {self.results_path}')

    def print_epoch(self, epoch):
        stats_dict = self.results[epoch - 1]
        epoch_results_str = fmt_stats_dict(stats_dict)
        print(f'Epoch: {epoch:02d}, {epoch_results_str}')

    def are_best_results(self, epoch_results, val_metric_key, less_is_more):
        if not self.best_epoch:
            return True
        if less_is_more:
            return self.results[self.best_epoch - 1][val_metric_key] >= epoch_results[val_metric_key]
        return self.results[self.best_epoch - 1][val_metric_key] <= epoch_results[val_metric_key]

    def evaluate(self, loader, split) -> dict:
        raise NotImplementedError

    def train(self) -> dict:
        return self.evaluate(self.train_loader, 'train')

    @torch.no_grad()
    def eval(self) -> dict:
        return self.evaluate(self.val_loader, 'val')

    @torch.no_grad()
    def test(self) -> dict:
        return self.evaluate(self.test_loader, 'test')


class SANETrainer(TorchModuleBaseTrainer):
    def __init__(self, model, model_nk, optimizer, optimizer_nk, train_loader, val_loader, loss_fn, loss_fn_nk, params):
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
        split_loss, split_loss_nk = 0.0, 0.0
        split_nle_loss, split_nle_loss_nk = 0.0, 0.0
        # Set split knowledge relevance indices
        split_ekri, split_pkri = 0.0, 0.0

        # Reset values for accuracy computation
        correct, correct_nk = 0, 0
        total = 0

        # Iterate over the DataLoader for training data
        pbar = tqdm(enumerate(dataloader, 0), total=len(dataloader))
        pbar.set_description(self.split_descriptions[split])
        start = time.time()
        for i, inputs in pbar:
            labels = inputs['gold_label'].to(settings.device)

            #####################################
            # (1) Compute loss without knowledge
            #####################################
            if train:
                self.optimizer_nk.zero_grad()

            # forward pass & compute loss without knowledge
            outputs = self.model_nk(inputs)
            pred_nk, nle_nk = outputs[:2]
            loss_nk = settings.alpha * nle_nk.loss.mean() + (1 - settings.alpha) * pred_nk.loss.mean()

            if train:
                # backward pass + optimization step
                loss_nk.backward()
                self.optimizer_nk.step()

            # Update Split Loss no knowledge
            split_loss_nk += loss_nk.item()
            # Update split nle loss no knowledge
            split_nle_loss_nk += nle_nk.loss.mean().item()
            # Update Accuracy
            predicted = pred_nk.logits.argmax(1)
            correct_nk += predicted.eq(labels).sum().item()
            # clean intermediate vars
            del outputs, predicted, loss_nk

            ##################################################
            # (2) Compute regret-augmented loss with knowledge
            ##################################################
            if train:
                # reset the gradients
                self.optimizer.zero_grad()

            # forward pass & compute loss
            outputs = self.model(inputs)
            pred, nle = outputs[:2]

            # Compute exact loss with knowledge
            loss = settings.alpha * nle.loss.mean() + (1 - settings.alpha) * pred.loss.mean()

            # Compute regret loss
            pred_regret = regret(pred.loss, pred_nk.loss.detach(), reduce=False)
            nle_regret = regret(nle.loss, nle_nk.loss.detach(), reduce=False)

            regret_loss = settings.alpha_regret * nle_regret.mean() + (1 - settings.alpha_regret) * pred_regret.mean()

            # Compute regret-augmented loss with knowledge
            augmented_loss = (1 - settings.beta) * loss + settings.beta * regret_loss

            if train:
                # backward pass + optimization step
                augmented_loss.backward()
                self.optimizer.step()

            # Update Split Loss
            split_loss += loss.item()
            # Update split nle loss
            split_nle_loss += nle.loss.mean().item()
            # Update Split Knowledge relevance
            split_ekri += nle.knowledge_relevance.mean().item()
            split_pkri += pred.knowledge_relevance.mean().item()
            # Update Accuracy
            predicted = pred.logits.argmax(1)
            correct += predicted.eq(labels).sum().item()

            # Update total
            total += len(labels)

        split_time = time.time() - start
        split_loss /= len(dataloader)
        split_loss_nk /= len(dataloader)
        split_ekri /= len(dataloader)
        split_pkri /= len(dataloader)
        split_acc = 100. * correct / total
        split_acc_nk = 100. * correct_nk / total

        return {
            f'{split}_acc': split_acc,
            f'{split}_acc_nk': split_acc_nk,
            f'{split}_loss': split_loss,
            f'{split}_loss_nk': split_loss_nk,
            f'{split}_nle_loss': split_nle_loss,
            f'{split}_nle_loss_nk': split_loss_nk,
            f'{split}_time': split_time,
            f'{split}_ekri': split_ekri,  # explanation knowledge relevance
            f'{split}_pkri': split_pkri,  # prediction knowledge relevance
        }


class SANENoKnowledgeTrainer(TorchModuleBaseTrainer):
    def __init__(self, model, optimizer, train_loader, val_loader, params):
        super().__init__(model, optimizer, train_loader, val_loader, params)

    def evaluate(self, dataloader, split):
        """Train, Evaluate, and Test the Model"""

        if dataloader is None:
            print(f'Skipping {self.split_descriptions[split]}')
            return {}

        train = split == 'train' and (not settings.frozen)

        self.model.train() if train else self.model.eval()

        train = split == 'train' and (not settings.frozen)

        self.model.train() if train else self.model.eval()

        # Set split loss value
        split_loss = 0.0
        split_nle_loss = 0.0

        # Reset values for accuracy computation
        correct = 0
        total = 0

        # Iterate over the DataLoader for training data
        pbar = tqdm(enumerate(dataloader, 0), total=len(dataloader))
        pbar.set_description(self.split_descriptions[split])
        start = time.time()
        for i, inputs in pbar:
            if train:
                # zero the parameter gradients
                self.optimizer.zero_grad()

            # forward pass & compute loss
            outputs = self.model(inputs)
            pred, nle = outputs[:2]

            # Compute model loss
            loss = settings.alpha * nle.loss + (1 - settings.alpha) * pred.loss

            if train:
                # backward pass + optimization step
                loss.backward()
                self.optimizer.step()

            # Update Split Loss
            split_loss += loss.item()
            # Update split nle loss
            split_nle_loss += nle.loss.item()
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
            f'{split}_nle_loss': split_nle_loss,
            f'{split}_time': split_time,
        }
