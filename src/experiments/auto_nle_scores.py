import os.path as osp

import pandas as pd
from bert_score import score as bert_score

from src.datasets.nl import get_dataset
from src.settings import settings
import evaluate


def compute_auto_nle_scores(results_path, test_dataset):
    # Load test results
    test_results_df = pd.read_csv(osp.join(results_path, f'test_results{settings.in_suffix}.csv'))

    assert len(test_results_df) == len(
        test_dataset), 'test_results and test_dataset must be of equal size, try checking data_frac.'

    # Prepare candidates sentences
    candidates = test_results_df['explanation'].tolist()

    # Prepare (multi-)reference sentences
    if 'Explanation_2' in test_dataset.nl and 'Explanation_3' in test_dataset.nl:
        multi_refs = [[exp1, exp2, exp3] for exp1, exp2, exp3 in
                      zip(test_dataset.nl['Explanation_1'], test_dataset.nl['Explanation_2'],
                          test_dataset.nl['Explanation_3'])]
    else:
        multi_refs = test_dataset.nl['Explanation_1']

    # Compute bert score
    (P, R, F), hashname = bert_score(candidates, multi_refs, lang='en', return_hash=True)
    bert_score_result = f'BERT_SCORE: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}\n'
    # Print bert score result
    print(bert_score_result)

    # Compute bleurt score
    bleurt = evaluate.load('bleurt', module_type='metric')
    bleurt_scores = bleurt.compute(predictions=candidates, references=multi_refs)['scores']
    bleurt_score_result = f'BLEURT: {sum(bleurt_scores)/len(bleurt_scores):.6f}\n'
    # Print bleurt score result
    print(bleurt_score_result)

    # Compute meteor score
    meteor = evaluate.load('meteor')
    meteor_scores = meteor.compute(predictions=candidates, references=multi_refs)['meteor']
    meteor_score_result = f'METEOR: {sum(meteor_scores)/len(meteor_scores):.6f}\n'
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
