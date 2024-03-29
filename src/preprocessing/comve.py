import os.path as osp

import pandas as pd

from src.preprocessing.common import preprocess
from src.settings import settings
from src.utils.embeddings import frozen_bart_tokenizer


def read_dataset():
    comve_dir = osp.join(settings.data_dir, 'comve')
    train_set = pd.read_csv(osp.join(comve_dir, 'train.csv'))
    val_set = pd.read_csv(osp.join(comve_dir, 'dev.csv'))
    test_set = pd.read_csv(osp.join(comve_dir, 'test.csv'))

    return {'train': train_set, 'val': val_set, 'test': test_set}


def reduce_dataset(splits, frac):
    for split, split_set in splits.items():
        # Replace Sentence1 and Sentence2 with Sentences
        sentences = []
        gold_labels = []
        sentence_pairs = zip(split_set['Correct Statement'], split_set['Incorrect Statement'])
        for i, (correct, incorrect) in enumerate(sentence_pairs):
            label = i % 2
            if label:
                sentence = correct + '. ' + frozen_bart_tokenizer.sep_token + ' ' + incorrect
            else:
                sentence = incorrect + '. ' + frozen_bart_tokenizer.sep_token + ' ' + correct

            # remove double periods
            sentence = sentence.replace('..', '.')

            sentences.append(sentence)
            gold_labels.append(label)

        split_set['Sentences'] = sentences
        split_set['gold_label'] = gold_labels
        # Drop useless columns
        useless_columns = ['Confusing Reason1', 'Confusing Reason2', 'Correct Statement', 'Incorrect Statement']
        split_set.drop(columns=useless_columns, inplace=True, errors='ignore')

        # Rename explanation columns
        split_set = split_set.rename(columns={f'Right Reason{i}': f'Explanation_{i}' for i in range(1, 4)})

        # Update split_set
        splits[split] = split_set

    return splits


if __name__ == '__main__':
    preprocess('comve', read_dataset, reduce_dataset, settings.data_frac)
