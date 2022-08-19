import os.path as osp

import pandas as pd
from bert_score import score

from src.datasets.esnli import get_dataset
from src.settings import settings


def compute_bert_score(results_path, test_dataset):
    # Load test results
    test_results_df = pd.read_csv(osp.join(results_path, f'test_results{settings.in_suffix}.csv'))

    assert len(test_results_df) == len(
        test_dataset), 'test_results and test_dataset must be of equal size, try checking data_frac.'

    # Prepare candidates and (multi-) refs
    candidates = test_results_df['explanation'].tolist()
    multi_refs = [[exp1, exp2, exp3] for exp1, exp2, exp3 in
                  zip(test_dataset.esnli['Explanation_1'], test_dataset.esnli['Explanation_2'],
                      test_dataset.esnli['Explanation_3'])]

    # Compute bert score
    (P, R, F), hashname = score(candidates, multi_refs, lang='en', return_hash=True)
    bert_score_result = f'{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}'

    # Print bert score result
    print(bert_score_result)

    # Save bert score result
    with open(osp.join(results_path, f'bert_score{settings.out_suffix}.txt'), 'w') as f:
        f.write(bert_score_result)


if __name__ == '__main__':
    # Get test dataloader
    test_dataset = get_dataset('test', 'esnli')
    results_path = settings.input_dir
    compute_bert_score(results_path, test_dataset)
