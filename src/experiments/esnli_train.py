import torch
import torch.optim as optim

from src.datasets.esnli import get_loader
from src.models.sane import SANE, SANENoKnowledge
from src.settings import settings
from src.utils.trainers import SANETrainer

# Create model
model = SANENoKnowledge() if settings.no_knowledge else SANE()
model = model.to(settings.device)

# Load expert
expert = None
if settings.expert is not None:
    expert = SANENoKnowledge().to(settings.device)
    model.load_state_dict(torch.load(settings.expert))
    model.eval()

# Train Model
SANETrainer(
    model=model,
    optimizer=optim.Adam(model.parameters(), lr=settings.lr),
    dataset_name='ESNLI',
    train_loader=get_loader('train'),
    val_loader=get_loader('val'),
    test_loader=get_loader('test') if settings.monitor_test else None,
    expert=expert,
).run()
