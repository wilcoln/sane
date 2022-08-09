import os.path as osp

import torch
from icecream import ic
from torch.utils.data import DataLoader

from src.datasets.esnli import ESNLIDataset, collate_fn
from src.experiments.esnli_test import test
from src.models.sane import SANE
from src.settings import settings

esnli_input_dict = {
    'Sentences': [
        'Wilfried is a student at the University of Basel.. He is a very good student.',
        'John is a professional football player.. He plays basketball for a living.',
        'SANE is a research project at the University of Basel.. It is a project to explain machine learning models.',
        'Github Copilot is an automated pair programming tool.. It is sentient',
        'A dog is running through long grass in a park-like setting.. A dog runs after a squirrel.',
        'A fireman searching for something using a flashlight.. A fireman is in the dark.',
    ],
    'gold_label': [
        2,
        0,
        2,
        1,
        2,
        1,
    ],
    'Explanation_1': [
        "We don't know whether Wilfried is a good student",
        'John cannot be a professional football player and plays basketball for a living',
        'Not all research projects are about explaining machine learning models',
        'An automated tool cannot be sentient',
        'The dog is not necessarily running after a squirrel',
        "he needs a flashlight because he's in the dark",
    ],
}

ic('Create Dataset')
dataset = ESNLIDataset(data_dict=esnli_input_dict)
ic('Create dataloader')
dataloader = DataLoader(dataset,
                      batch_size=settings.batch_size, shuffle=False,
                      num_workers=settings.num_workers,
                      collate_fn=collate_fn)

# dataloader = get_loader('test')

# Load model
ic('Load Model')
input_dir = 'results/trainers/2022-08-08_22-41-59_516527_dataset=ESNLI_model=KAX_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_num_attn_heads=2'
model = SANE().to(settings.device)
model.load_state_dict(torch.load(osp.join(input_dir, 'model.pt')))
model.eval()

results_path = settings.results_dir

# test model
ic('Test')
test(model, results_path, dataloader)
