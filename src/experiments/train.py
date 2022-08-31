import torch.optim as optim
from torch import nn

from src.datasets.nl import get_loader
from src.experiments.suite import run_suite
from src.models.sane import SANE, SANENoKnowledge
from src.settings import settings
from src.utils.trainers import SANETrainer, SANENoKnowledgeTrainer

if not settings.no_knowledge:
    # Create models
    model = SANE().to(settings.device)
    model_nk = SANENoKnowledge().to(settings.device) if not settings.no_train_nk else None

    # Create optimizers
    optimizer = optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    optimizer_nk = optim.AdamW(model_nk.parameters(), lr=settings.lr,
                               weight_decay=settings.weight_decay) if not settings.no_train_nk else None

    # Train Model
    trainer = SANETrainer(
        model=model,
        model_nk=model_nk,
        optimizer=optimizer,
        optimizer_nk=optimizer_nk,
        train_loader=get_loader('train'),
        val_loader=get_loader('val'),
        params=settings.exp,
    )
else:
    # Create models
    model = SANENoKnowledge().to(settings.device)

    # Create optimizers
    optimizer = optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)

    # Train Model
    trainer = SANENoKnowledgeTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=get_loader('train'),
        val_loader=get_loader('val'),
        params=settings.exp,
    )

trainer.run()

# Run experiments
results_path = trainer.results_path
run_suite(results_path)
