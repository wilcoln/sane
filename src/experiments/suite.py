import torch

from src.datasets.nl import get_dataset, get_loader
from src.experiments.auto_nle_eval import compute_auto_nle_scores
from src.experiments.knowledge_attention_map import compute_knowledge_attention_map, plot_knowledge_attention_map
from src.experiments.knowledge_indices import compute_knowledge_indices
from src.experiments.test import test
from src.models.sane import SANE
from src.settings import settings
import os.path as osp


def run_suite(results_path):
    # Setup model and loader
    dataloader = get_loader('test')
    model = SANE()
    model.load_state_dict(torch.load(osp.join(results_path, 'model.pt')))
    model = model.to(settings.device)
    model.eval()

    # Run experiments
    # 1. Test
    print('Running test...')
    test(model, results_path, dataloader)
    # 2. Auto NLE Evaluation
    print('Running auto NLE evaluation...')
    try:
        compute_auto_nle_scores(results_path)
    except Exception as e:
        print(e)
        print('Auto NLE evaluation Failed')
    # 3. Knowledge Attention Map
    print('Running knowledge attention map...')
    compute_knowledge_attention_map(model, results_path)
    plot_knowledge_attention_map(results_path)
    # 4. Knowledge Indices
    print('Running knowledge indices...')
    # Load model
    inputs = next(iter(dataloader))
    compute_knowledge_indices(model, inputs, results_path)
    print('Done.')


if __name__ == '__main__':
    results_path = settings.input_dir
    run_suite(results_path)
