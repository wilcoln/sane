import os.path as osp

import torch
from icecream import ic
from torch.utils.data import DataLoader

from src.datasets.nl import NLDataset, collate_fn
from src.experiments.esnli_test import test
from src.models.sane import SANE, SANENoKnowledge
from src.settings import settings

esnli_input_dict = {
    'Sentences': [
        'Wilfried is a student at the University of Basel.. He is from Cameroon',
        'John is a professional football player.. He plays basketball for a living.',
        'SANE is a research project at the University of Basel.. It is a project on AI explanability',
        'SANE beats state-of-the-art performance on ESNLI dataset by a large margin.. SANE must be bugged',
        'Github Copilot is an automated pair programming tool.. It is sentient',
        'A dog is running through long grass in a park-like setting.. A dog runs after a squirrel.',
        'A fireman searching for something using a flashlight.. A fireman is in the dark.',
    ],
    'gold_label': [
        2,
        0,
        0,
        2,
        0,
        2,
        1,
    ],
    'Explanation_1': [
        "We don't know whether Wilfried is from Cameroon",
        'John cannot be a professional football player and plays basketball for a living',
        'Not all research projects are about explaining machine learning models',
        "Just because SANE beats sota performance doesn't mean it is bugged",
        'An automated tool cannot be sentient',
        'The dog is not necessarily running after a squirrel',
        "he needs a flashlight because he's in the dark",
    ],
}

ic('Create Dataset')
dataset = NLDataset(data_dict=esnli_input_dict)
ic('Create dataloader')
dataloader = DataLoader(dataset,
                        batch_size=settings.batch_size, shuffle=False,
                        num_workers=settings.num_workers,
                        collate_fn=collate_fn)

# dataloader = get_loader('test')

# Load model
ic('Load Model')
input_dir = settings.input_dir
model = SANENoKnowledge() if settings.no_knowledge else SANE()
model = model.to(settings.device)
model.load_state_dict(torch.load(osp.join(input_dir, 'model.pt')))
model.eval()

results_path = osp.join(settings.results_dir, 'custom')

# test model
ic('Test')
test(model, results_path, dataloader)
