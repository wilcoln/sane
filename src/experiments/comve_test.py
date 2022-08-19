import os.path as osp
import torch

from src.datasets.comve import get_loader
from src.experiments.esnli_test import test
from src.models.sane import SANE, SANENoKnowledge
from src.settings import settings

if __name__ == '__main__':
    # Get test dataloader
    dataloader = get_loader('test')

    # Load model
    model = SANENoKnowledge() if settings.no_knowledge else SANE()
    model = model.to(settings.device)
    results_path = settings.input_dir
    model.load_state_dict(torch.load(osp.join(results_path, 'model.pt')))
    model.eval()

    # test model
    test(model, results_path, dataloader)
