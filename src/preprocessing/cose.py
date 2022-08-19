import json
import os.path as osp

import numpy as np
import pandas as pd
from icecream import ic

from src.preprocessing.common import preprocess
from src.settings import settings


def jsonl_to_dataframe(jsonl_file):
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
        extracted_data = pd.json_normalize(data)
        extracted_data.columns = extracted_data.columns.map(lambda x: x.split('.')[-1])

    return extracted_data


def add_choices_and_label_columns(extracted_data):
    num_choices = len(extracted_data['choices'][0])

    def get_choices(options, val):
        option = options[int(val)]
        return option.get('text')

    choices = np.arange(num_choices)
    choices = choices.astype('str')
    for c in choices:
        extracted_data['choice_' + c] = extracted_data['choices'].apply(lambda x: get_choices(x, c))
    answer_match = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    def get_label(answer):
        return answer_match.get(answer)

    extracted_data['label'] = extracted_data['answerKey'].apply(lambda x: get_label(x))

    return extracted_data


def read_dataset():
    cose_dir = osp.join(settings.data_dir, 'cose')
    train_set = jsonl_to_dataframe(osp.join(cose_dir, 'train_rand_split.jsonl'))
    train_set = add_choices_and_label_columns(train_set)
    e_train_set = jsonl_to_dataframe(osp.join(cose_dir, 'cose_train_v1.11_processed.jsonl'))
    train_set = train_set.set_index('id').join(e_train_set.set_index('id'))

    val_set = jsonl_to_dataframe(osp.join(cose_dir, 'dev_rand_split.jsonl'))
    val_set = add_choices_and_label_columns(val_set)
    e_val_set = jsonl_to_dataframe(osp.join(cose_dir, 'cose_dev_v1.11_processed.jsonl'))
    val_set = val_set.set_index('id').join(e_val_set.set_index('id'))

    return {'train': train_set, 'val': val_set}


def reduce_dataset(splits, frac):
    for split, split_set in splits.items():
        # Concatenate question and choice_i columns into one column 'Sentences'
        split_set['Sentences'] = split_set['stem'] + '? A) ' + split_set['choice_0'] + ', B) ' + split_set[
            'choice_1'] + ', C) ' + split_set['choice_2'] + ', D) ' + split_set['choice_3'] + ', E) ' + split_set[
                                     'choice_4'] + '.'
        split_set['Sentences'] = split_set['Sentences'].str.replace('??', '?', regex=False)
        split_set['Sentences'] = split_set['Sentences'].str.replace('..', '.', regex=False)

        # Drop useless columns
        useless_columns = [
            'stem', 'answerKey', 'selected', 'question_concept', 'choices',
            'choice_0', 'choice_1', 'choice_2', 'choice_3', 'choice_4',
        ]
        split_set.drop(columns=useless_columns, inplace=True, errors='ignore')
        # Rename explanation columns
        split_set = split_set.rename(columns={'open-ended': 'Explanation_1', 'label': 'gold_label'})
        # Update split_set
        splits[split] = split_set
    return splits


if __name__ == '__main__':
    preprocess('cose', read_dataset, reduce_dataset, settings.data_frac)
