from torch.utils.data import DataLoader
from icecream import ic
from torch import nn
from tqdm import tqdm

from utils.settings import settings
from datasets.esnli import ESNLIDataset
from models.kax import KAX
import os.path as osp

# Load dataset splits
esnli_dir = osp.join(settings.data_dir, 'esnli')
train_set = ESNLIDataset(path=esnli_dir, split='train', frac=.01)
val_set = ESNLIDataset(path=esnli_dir, split='val', frac=.01)
test_set = ESNLIDataset(path=esnli_dir, split='test', frac=.01)

# Create Loaders

train_loader = DataLoader(train_set, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers)
val_loader = DataLoader(val_set, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers)
test_loader = DataLoader(test_set, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers)

# Define model
model = KAX().to(settings.device)

# Define loss function and optimizer
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the network
from collections import defaultdict

losses = defaultdict(list)
accuracies = defaultdict(list)


# Definition of the Training/Evaluation Function
def evaluate(dataloader, split):
    assert split in {'train', 'val', 'test'}, "split must be either 'train', 'val' or 'test'"

    train = split == 'train'

    model.train() if train else model.eval()

    # Set current loss value
    current_loss = 0.0

    # Reset values for accuracy computation
    correct = 0
    total = 0

    # Iterate over the DataLoader for training data
    for i, inputs in tqdm(list(enumerate(dataloader, 0))):
        inputs['gold_label'] = inputs['gold_label'].long().to(settings.device)

        if train:
            # zero the parameter gradients
            optimizer.zero_grad()

        # forward pass & compute loss
        (nles, outputs), loss = model(inputs)

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

    current_loss /= len(dataloader)
    current_accuracy = 100. * correct / total

    losses[split].append(current_loss)
    accuracies[split].append(current_accuracy)

    print('%s Loss: %.3f | %s Acc: %.1f' % (split.capitalize(), current_loss, split.capitalize(), current_accuracy))


train = lambda: evaluate(train_loader, split='train')
valid = lambda: evaluate(val_loader, split='val')
# test = lambda : evaluate(testloader, split='test')


# Train & Evaluate
for epoch in range(settings.num_epochs):  # loop over the dataset multiple times
    print(f'---\nEpoch {epoch + 1}')
    train()
    valid()

print('---\nFinished Training.')

# Plot Learning curves (Loss & Accuracy)
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(16, 6)

ax1.plot(losses['train'], '-o')
ax1.plot(losses['val'], '-o')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Losses')
ax1.set_title('Train vs Valid Losses')
ax1.legend(['Train', 'Valid'])

ax2.plot(accuracies['train'], '-o')
ax2.plot(accuracies['val'], '-o')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracies')
ax2.set_title('Train vs Valid Accuracies')
ax2.legend(['Train', 'Valid'])

plt.show()
