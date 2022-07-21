import time

import math
import torch
from icecream import ic
from torch.utils.data import DataLoader, default_collate
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data.hetero_data import HeteroData as PygData
from utils.trainers import TorchModuleBaseTrainer
from utils.settings import settings
from datasets.esnli import ESNLIDataset
from models.kax import KAX
import os.path as osp

# Load dataset splits
original_sizes = {'train': 549367, 'val': 9842, 'test': 9824}

# Custom collate fn
def collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    return {key: (default_collate([d[key] for d in batch]) if key != 'pyg_data' else [d[key] for d in batch] ) for key in elem}


def get_loaders(split):
    split_size = int(original_sizes[split]*settings.data_frac)
    return [
        DataLoader(ESNLIDataset(path=settings.data_dir, split=split, frac=settings.data_frac, chunk=chunk), batch_size=32, shuffle=False, num_workers=settings.num_workers, collate_fn=collate_fn)
        for chunk in range(math.ceil(split_size/settings.chunk_size))
    ]

# Create Loaders
train_loaders = get_loaders('train')
val_loaders = get_loaders('val')
test_loaders = get_loaders('test')

sample_train_set = ESNLIDataset(path=settings.data_dir, frac=settings.data_frac, split='train', chunk=0)
# Define model
model = KAX(metadata=sample_train_set[0]['pyg_data'].metadata()).to(settings.device)

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the network
losses = defaultdict(list)
accuracies = defaultdict(list)


class KAXTrainer(TorchModuleBaseTrainer):
    def __init__(self, model, optimizer, dataset_name, train_loaders, val_loaders, test_loaders):
        super().__init__(model, optimizer, settings.num_epochs, dataset_name)
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders

    def evaluate(self, dataloaders, split):
        """Train, Evaluate, and Test the Model"""
        assert split in {'train', 'val', 'test'}, "split must be either 'train', 'val' or 'test'"

        train = split == 'train'

        self.model.train() if train else self.model.eval()

        # Set current loss value
        current_loss = 0.0

        # Reset values for accuracy computation
        correct = 0
        total = 0

        start = time.time()
        pbar = tqdm(dataloaders)
        description = {'train': 'Training', 'val': 'Validation', 'test': 'Testing'}
        pbar.set_description(description[split])
        for dataloader in pbar:
            # Iterate over the DataLoader for training data
            for i, inputs in tqdm(list(enumerate(dataloader, 0))):
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(settings.device)

                if train:
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                # forward pass & compute loss
                nles, outputs, loss = self.model(inputs)

                if train:
                    # backward pass + optimization step
                    loss.backward()
                    self.optimizer.step()

                # Update Loss
                current_loss += loss.item()

                # Update Accuracy
                _, predicted = outputs.max(1)
                total += inputs['gold_label'].size(0)
                correct += predicted.eq(inputs['gold_label']).sum().item()
        
        split_time = time.time() - start
        current_loss /= sum(len(dl) for dl in dataloaders)
        current_acc = 100. * correct / total

        return {
            f'{split}_acc': current_acc,
            f'{split}_loss': current_loss,
            f'{split}_time': split_time
        }

    def train(self) -> dict:
        return self.evaluate(self.train_loaders, 'train')

    def eval(self) -> dict:
        return self.evaluate(self.val_loaders, 'val')

    def test(self) -> dict:
        return self.evaluate(self.test_loaders, 'test')


KAXTrainer(model, optimizer, sample_train_set.name, train_loaders, val_loaders, test_loaders).run()
