import pandas as pd
import numpy as np
from icecream import ic

from src.settings import settings
import os.path as osp
# Load esnli_human eval for rexc and sane in two different dataframes
rexc_df = pd.read_csv(osp.join(settings.data_dir, 'esnli_human_eval_rexc.csv'))
sane_df = pd.read_csv(osp.join(settings.data_dir, 'esnli_human_eval_sane.csv'))
# rename explanation column to SANE
sane_df.rename(columns={'explanation': 'SANE'}, inplace=True)
# drop sane's rows where the prediction is not equal to the gold label
sane_df = sane_df[sane_df['prediction'] == sane_df['gold_label']]

# drop sane's gold label column
sane_df.drop(columns=['gold_label'], inplace=True)

# add sentence column to rexc_df, which is the concatenation of premise and hypothesis
rexc_df['sentence'] = rexc_df['Sentence1'] + '. ' + rexc_df['Sentence2']

# Create a new dataframe that is the intersection of rexc and sane on the sentence1 and sentence2 columns
df = pd.merge(rexc_df, sane_df, on=['sentence'], how='inner')

# Keep only columns sentence, SANE, RExC
df = df[['sentence', 'gold_label', 'SANE', 'RExC']]

# Sample n bernoulli variables where is n is the length the dataframe
sane_index = np.random.binomial(1, 0.5, size=len(df))

# Create a new column model_1 that is sane if sane_index is 0 and rexc if sane_index is 1
df['model_1'] = df['SANE'].where(sane_index == 0, df['RExC'])
# Create a new column model_2 that is rexc if sane_index is 0 and sane if sane_index is 1
df['model_2'] = df['SANE'].where(sane_index == 1, df['RExC'])

# Drop the columns SANE and RExC
df.drop(columns=['SANE', 'RExC'], inplace=True)

# Save the new dataframe to a csv file
df.to_csv(osp.join(settings.data_dir, 'esnli_human_eval_sane_rexc.csv'), index=False)

# Save sane_index to a csv file
pd.DataFrame(sane_index).to_csv(osp.join(settings.data_dir, 'esnli_human_eval_sane_index.csv'), index=False)
