import math
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate, ConcatDataset
from tqdm import tqdm

from src.datasets.esnli import ESNLIDataset
from src.models.kax import KAX
from src.settings import settings
from src.utils.embeddings import tokenize
from src.utils.trainers import TorchModuleBaseTrainer

# Load dataset splits
og_sizes = {'train': 549367, 'val': 9842, 'test': 9824}
new_sizes = {split: int(og_size * settings.data_frac) for split, og_size in og_sizes.items()}
num_chunks = {split: math.ceil(new_size / settings.chunk_size) for split, new_size in new_sizes.items()}


# Custom collate fn
def collate_fn(batch):
    elem = batch[0]

    def collate(key):
        if key in {'Sentences', 'Explanation_1'}:
            return tokenize([d[key] for d in batch])
        if key in {'concept_ids'}:
            return [torch.LongTensor(d[key]) for d in batch]
        return default_collate([d[key] for d in batch])

    return {key: collate(key) for key in elem}


def get_loader(split):
    datasets = [ESNLIDataset(path=settings.data_dir, split=split, frac=settings.data_frac, chunk=chunk)
                for chunk in range(num_chunks[split])]

    return DataLoader(ConcatDataset(datasets),
                      batch_size=settings.batch_size, shuffle=False,
                      num_workers=settings.num_workers,
                      collate_fn=collate_fn)


# Create Loaders
train_loader = get_loader('train')
val_loader = get_loader('val')
test_loader = get_loader('test')

# Define model
model = KAX().to(settings.device)
dataset_name = 'ESNLI'

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)


class KAXTrainer(TorchModuleBaseTrainer):
    def __init__(self, model, optimizer, num_epochs, dataset_name, train_loader, val_loader, test_loader):
        super().__init__(model, optimizer, num_epochs, dataset_name)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def evaluate(self, dataloader, split):
        """Train, Evaluate, and Test the Model"""
        assert split in {'train', 'val', 'test'}, "split must be either 'train', 'val' or 'test'"

        train = split == 'train'

        self.model.train() if train else self.model.eval()

        # Set current loss value
        current_loss = 0.0

        # Reset values for accuracy computation
        correct = 0
        total = 0

        # Iterate over the DataLoader for training data
        pbar = tqdm(enumerate(dataloader, 0), total=len(dataloader))
        description = {'train': 'Training', 'val': 'Validation', 'test': 'Testing'}
        pbar.set_description(description[split])
        start = time.time()
        for i, inputs in pbar:
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to(settings.device)

            if train:
                # zero the parameter gradients
                self.optimizer.zero_grad()

            # forward pass & compute loss
            attns, nles_tokens, nles, outputs, loss = self.model(inputs)

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
        current_loss /= len(dataloader)
        current_acc = 100. * correct / total

        return {
            f'{split}_acc': current_acc,
            f'{split}_loss': current_loss,
            f'{split}_time': split_time,
        }

    def train(self) -> dict:
        return self.evaluate(self.train_loader, 'train')

    def eval(self) -> dict:
        return self.evaluate(self.val_loader, 'val')

    def test(self) -> dict:
        return self.evaluate(self.test_loader, 'test')


KAXTrainer(model, optimizer, settings.num_epochs, dataset_name, train_loader, val_loader, test_loader).run()
