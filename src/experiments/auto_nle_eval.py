import os.path as osp

import evaluate
import pandas as pd
import torch
from bert_score import score as bert_score
from bleurt import score as bleurt_score

from src.datasets.nl import get_dataset
from src.settings import settings


def compute_auto_nle_scores(results_path):
    # Check if results already exist
    txt_path = osp.join(results_path, f'auto_nle_eval.txt')
    if not settings.overwrite and osp.exists(txt_path):
        return

    # Load test results
    test_dataset = get_dataset('test', settings.dataset)
    test_results_df = pd.read_csv(osp.join(results_path, f'test_results.csv'))

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

    # Compute bleurt score
    bleurt_scorer = bleurt_score.BleurtScorer()
    bleurt_scores = [
        bleurt_scorer.score(candidates=candidates, references=get_explanations(i + 1))
        for i in range(num_refs)
    ]
    bleurt_scores = torch.Tensor(bleurt_scores).max(0)[0]
    bleurt_score_result = f'BLEURT: {bleurt_scores.mean().item():.6f}\n'

    # Compute meteor score
    meteor = evaluate.load('meteor', module_type='metric')
    meteor_scores = [
        meteor.compute(predictions=candidates, references=get_explanations(i + 1))['meteor']
        for i in range(num_refs)
    ]
    meteor_scores = torch.Tensor(meteor_scores).max(0)[0]
    meteor_score_result = f'METEOR: {meteor_scores.mean().item():.6f}\n'

    # Print auto nle scores
    print(bert_score_result, bleurt_score_result, meteor_score_result)

    # Save bert score result
    with open(txt_path, 'w') as f:
        f.write(bert_score_result)
        f.write(bleurt_score_result)
        f.write(meteor_score_result)


if __name__ == '__main__':
    # Get test dataloader
    results_path = settings.input_dir
    compute_auto_nle_scores(results_path)
