import torch.optim as optim
from torch import nn

from src.datasets.nl import get_loader
from src.models.sane import SANE, SANENoKnowledge
from src.settings import settings
from src.utils.trainers import SANETrainer

# Create model
model = SANE().to(settings.device)
model_nk = SANENoKnowledge().to(settings.device)

# Train Model
SANETrainer(
    model=model,
    model_nk=model_nk,
    optimizer=optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay),
    optimizer_nk=optim.AdamW(model_nk.parameters(), lr=settings.lr, weight_decay=settings.weight_decay),
    loss_fn=nn.CrossEntropyLoss(),
    loss_fn_nk=nn.CrossEntropyLoss(),
    train_loader=get_loader('train'),
    val_loader=get_loader('val'),
    params=settings.exp,
).run()
