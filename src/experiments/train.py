import torch.optim as optim

from src.datasets.nl import get_loader
from src.models.sane import SANE, SANENoKnowledge
from src.settings import settings
from src.utils.trainers import SANETrainer

# Create model
model = SANENoKnowledge() if settings.no_knowledge else SANE()
model = model.to(settings.device)

# Train Model
SANETrainer(
    model=model,
    optimizer=optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay),
    train_loader=get_loader('train'),
    val_loader=get_loader('val'),
).run()
