import math

import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from src.datasets.esnli import ESNLIDataset, collate_fn as esnli_collate_fn
from src.models.kax import KAX
from src.settings import settings
from src.utils.trainers import KAXTrainer

# Load dataset splits
og_sizes = {'train': 549367, 'val': 9842, 'test': 9824}
new_sizes = {split: int(og_size * settings.data_frac) for split, og_size in og_sizes.items()}
num_chunks = {split: math.ceil(new_size / settings.chunk_size) for split, new_size in new_sizes.items()}


def get_loader(split):
    datasets = [
        ESNLIDataset(path=settings.data_dir, split=split, frac=settings.data_frac, chunk=chunk)
        for chunk in range(num_chunks[split])
    ]

    return DataLoader(ConcatDataset(datasets),
                      batch_size=settings.batch_size, shuffle=False,
                      num_workers=settings.num_workers,
                      collate_fn=esnli_collate_fn)


# Create model
model = KAX().to(settings.device)

# Train Model
KAXTrainer(
    model=model,
    optimizer=optim.Adam(model.parameters(), lr=settings.lr),
    dataset_name='ESNLI',
    train_loader=get_loader('train'),
    val_loader=get_loader('val'),
    test_loader=get_loader('test'),
).run()
