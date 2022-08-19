import os.path as osp

import torch
import torch.optim as optim

from src.datasets.nl import get_loader
from src.models.sane import SANE, SANENoKnowledge
from src.settings import settings
from src.utils.nn import freeze
from src.utils.trainers import SANETrainer

# Create model
model = SANENoKnowledge() if settings.no_knowledge else SANE()
model = model.to(settings.device)

# Load expert
expert = None
if settings.expert is not None:
    expert = SANENoKnowledge().to(settings.device)
    expert.load_state_dict(torch.load(osp.join(settings.expert, 'model.pt')))
    expert = freeze(expert)

# Train Model
SANETrainer(
    model=model,
    optimizer=optim.Adam(model.parameters(), lr=settings.lr),
    train_loader=get_loader('train'),
    val_loader=get_loader('val'),
    expert=expert,
).run()
