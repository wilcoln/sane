import os.path as osp
import torch
import pandas as pd
from bert_score import score as bert_score

from src.datasets.nl import get_dataset
from src.settings import settings
import evaluate
from icecream import ic

def compute_auto_nle_scores(results_path, test_dataset):
    # Load test results
    test_results_df = pd.read_csv(osp.join(results_path, f'test_results{settings.in_suffix}.csv'))

    assert len(test_results_df) == len(
        test_dataset), 'test_results and test_dataset must be of equal size, try checking data_frac.'

    # Prepare candidates sentences
    candidates = test_results_df['explanation'].tolist()

    # Prepare (multi-)reference sentences
    keys = set(test_dataset.datasets[0].nl.keys())

    def get_explanations(num):
        return [d[f'Explanation_{num}'] for d in test_dataset]

    num_refs = 3 if ('Explanation_2' in keys and 'Explanation_3') in keys else 1
    if num_refs == 3:
        refs = list(zip(get_explanations(1), get_explanations(2), get_explanations(3)))
    else:
        refs = [get_explanations(1)]

    # Compute bert score
    (P, R, F), hashname = bert_score(candidates, refs, lang='en', return_hash=True)
    bert_score_result = f'BERT_SCORE: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}\n'
    # Print bert score result
    print(bert_score_result)


    def get_metric_results(metric):
        metric = evaluate.load(metric, module_type='metric')
        metric_scores = []
        for i in range(num_refs):
            metric_score = metric.compute(predictions=candidates, references=get_explanations(i+1))['scores']
            metric_scores.append(metric_score)

        metric_scores = torch.Tensor(metric_scores).max(0)[0]
        ic(metric_scores)
        return f'{metric.upper()}: {metric_scores.mean().item():.6f}\n'

    # Compute bleurt score
    bleurt_score_result = get_metric_results('bleurt')
    # Print bleurt score result
    print(bleurt_score_result)

    # Compute meteor score
    meteor_score_result = get_metric_results('meteor')
    # Print meteor score result
    print(meteor_score_result)

    # Save bert score result
    with open(osp.join(results_path, f'auto_nle_scores{settings.out_suffix}.txt'), 'w') as f:
        f.write(bert_score_result)
        f.write(bleurt_score_result)
        f.write(meteor_score_result)


if __name__ == '__main__':
    # Get test dataloader
    test_dataset = get_dataset('test', settings.dataset)
    results_path = settings.input_dir
    compute_auto_nle_scores(results_path, test_dataset)
