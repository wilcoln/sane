import time

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
esnli_dir = osp.join(settings.data_dir, 'esnli', 'toy')
train_set = ESNLIDataset(path=esnli_dir, split='train')
val_set = ESNLIDataset(path=esnli_dir, split='val')
test_set = ESNLIDataset(path=esnli_dir, split='test')

# Custom collate fn : TODO
def collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    return {key: (default_collate([d[key] for d in batch]) if key != 'pyg_data' else [d[key] for d in batch] ) for key in elem}

# Create Loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=settings.num_workers, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=settings.num_workers, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=settings.num_workers, collate_fn=collate_fn)

# Define model
model = KAX(metadata=train_set[0]['pyg_data'].metadata()).to(settings.device)

# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the network
losses = defaultdict(list)
accuracies = defaultdict(list)


class KAXTrainer(TorchModuleBaseTrainer):
    def __init__(self, model, optimizer, train_loader, val_loader, test_loader):
        super().__init__(model, optimizer, settings.num_epochs, train_loader, val_loader, test_loader)

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

        start = time.time()
        # Iterate over the DataLoader for training data
        pbar = tqdm(list(enumerate(dataloader, 0)))
        description = {'train': 'Training', 'val': 'Validation', 'test': 'Testing'}
        pbar.set_description(description[split])
        for i, inputs in pbar:
            inputs['gold_label'] = inputs['gold_label'].long().to(settings.device)

            if train:
                # zero the parameter gradients
                optimizer.zero_grad()

            # forward pass & compute loss
            nles, outputs, loss = self.model(inputs)

            if train:
                # backward pass + optimization step
                loss.backward()
                optimizer.step()

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
            f'{split}_time': split_time
        }

    def train(self) -> dict:
        return self.evaluate(self.train_loader, 'train')

    def eval(self) -> dict:
        return self.evaluate(self.val_loader, 'val')

    def test(self) -> dict:
        return self.evaluate(self.test_loader, 'test')


KAXTrainer(model, optimizer, train_loader, val_loader, test_loader).run()
