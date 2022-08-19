from src.datasets.esnli import get_dataset
from src.experiments.esnli_bert_score import compute_bert_score
from src.settings import settings


if __name__ == '__main__':
    # Get test dataloader
    test_dataset = get_dataset('test', 'comve')
    results_path = settings.input_dir
    compute_bert_score(results_path, test_dataset)
