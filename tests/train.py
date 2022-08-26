import torch.optim as optim

from src.datasets.nl import get_sanity_check_loader
from src.models.sane import SANE, SANENoKnowledge
from src.settings import settings
from src.utils.trainers import SANETrainer

# Create model
model = SANE().to(settings.device)
model_nk = SANENoKnowledge().to(settings.device)
dataloader = get_sanity_check_loader()
# Train Model
SANETrainer(
    model=model,
    model_nk=model_nk,
    optimizer=optim.Adam(model.parameters(), lr=settings.lr),
    optimizer_nk=optim.Adam(model_nk.parameters(), lr=settings.lr),
    train_loader=dataloader,
    val_loader=dataloader,
).run()
