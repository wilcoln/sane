import torch.optim as optim

from src.datasets.esnli import get_sanity_check_loader
from src.models.sane import SANE
from src.settings import settings
from src.utils.trainers import SANETrainer

# Create model
model = SANE().to(settings.device)
dataloader = get_sanity_check_loader()
# Train Model
SANETrainer(
    model=model,
    optimizer=optim.Adam(model.parameters(), lr=settings.lr),
    dataset_name='ESNLI',
    train_loader=dataloader,
    val_loader=dataloader,
).run()
