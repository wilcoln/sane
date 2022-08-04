import os.path as osp

import torch
from icecream import ic
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from datasets.esnli import ESNLIDataset, conceptnet, tokenizer
from models.kax import KAXWK
from ..utils.settings import settings


def collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    return {key: (default_collate([d[key] for d in batch]) if key != 'pyg_data' else [d[key] for d in batch]) for key in
            elem}


# Load test set
dataloader = DataLoader(ESNLIDataset(path=settings.data_dir, split='test', frac=.25, chunk=0),
                        batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers,
                        collate_fn=collate_fn)

# Load model
model = KAXWK(metadata=conceptnet.metadata()).to(settings.device)
model_path = osp.join(settings.results_dir,
                      'trainers/2022-08-03_20-33-55_636021_dataset=esnli_train_0_model=KAXWK_num_epochs=5/model.pt')
model.load_state_dict(torch.load(model_path))
model.eval()

# Run model on test set
for i, inputs in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
    for k in inputs:
        if isinstance(inputs[k], torch.Tensor):
            inputs[k] = inputs[k].to(settings.device)

    # forward pass & compute loss
    attns, nles_tokens, nles, outputs, loss = model(inputs)
    # display explanations
    ic(tokenizer.batch_decode(nles_tokens, skip_special_tokens=True))
