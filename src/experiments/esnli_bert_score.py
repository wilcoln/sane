import pandas as pd
from tqdm import tqdm
import os.path as osp

from src.datasets.esnli import ESNLIDataset
from src.settings import settings
from bert_score import score


results_path = 'results/trainers/2022-08-07_00-06-16_306263_dataset=ESNLI_model=KAX_num_epochs=5_batch_size=80_lr=0.0001_sent_dim=768_hidden_dim=32_nle_pred=True_max_concepts_per_sent=200_sentence_pool=mean_data_frac=1.0_alpha=0.4'

# Load test results
test_results_df = pd.read_csv(osp.join(results_path, 'test_results.csv'))

# Load test dataset (to access gold explanations)
test_dataset = ESNLIDataset(path=settings.data_dir, split='test', frac=settings.data_frac, chunk=0)


assert len(test_results_df) == len(test_dataset), 'test_results and test_dataset must be of equal size, try checking data_frac.'

# Prepare candidates and (multi-) refs
cands = test_results_df['explanation'].tolist()
multi_refs = [[exp1, exp2, exp3] for exp1, exp2, exp3 in zip(test_dataset.esnli['Explanation_1'], test_dataset.esnli['Explanation_2'], test_dataset.esnli['Explanation_3'])]

# Compute bert score
(P, R, F), hashname = score(cands, multi_refs, lang='en', return_hash=True)
bert_score_result = f'{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}'

# Print bert score result
print(bert_score_result)

# Save bert score result
with open(osp.join(results_path, 'bert_score.txt'), 'w') as f:
    f.write(bert_score_result)
