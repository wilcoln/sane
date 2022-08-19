
import os.path as osp

import pandas as pd

from src.preprocessing.common import preprocess
from src.settings import settings


def read_dataset():
    esnli_dir = osp.join(settings.data_dir, 'esnli')
    train_set_1 = pd.read_csv(osp.join(esnli_dir, 'esnli_train_1.csv'))
    train_set_2 = pd.read_csv(osp.join(esnli_dir, 'esnli_train_2.csv'))
    train_set = pd.concat([train_set_1, train_set_2], ignore_index=True)

    val_set = pd.read_csv(osp.join(esnli_dir, 'esnli_dev.csv'))

    test_set = pd.read_csv(osp.join(esnli_dir, 'esnli_test.csv'))

    return {'train': train_set, 'val': val_set, 'test': test_set}


def reduce_dataset(splits, frac):
    for split, split_set in splits.items():
        split_set = split_set.sample(int(len(split_set) * frac), random_state=0)
        # Replace Sentence1 and Sentence2 with Sentences
        split_set['Sentences'] = split_set['Sentence1'] + '. ' + split_set['Sentence2']
        split_set['Sentences'] = split_set['Sentences'].str.replace('..', '.', regex=False)
        # Drop useless columns
        useless_columns = [
            'pairID', 'WorkerId',
            'Sentence1', 'Sentence2',
            'Sentence1_marked_1', 'Sentence2_marked_1',
            'Sentence1_Highlighted_1', 'Sentence2_Highlighted_1',
            'Sentence1_marked_2', 'Sentence2_marked_2',
            'Sentence1_Highlighted_2', 'Sentence2_Highlighted_2',
            'Sentence1_marked_3', 'Sentence2_marked_3',
            'Sentence1_Highlighted_3', 'Sentence2_Highlighted_3'
        ]
        split_set.drop(columns=useless_columns, inplace=True, errors='ignore')
        # Convert labels to categorical
        split_set['gold_label'] = split_set['gold_label'].astype('category').cat.codes
        # Update split_set
        splits[split] = split_set

    return splits


if __name__ == '__main__':
    preprocess('esnli', read_dataset, reduce_dataset, data_frac=settings.data_frac)
