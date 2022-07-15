import pandas as pd

import os

from icecream import ic

from utils.embeddings import bart
from utils.settings import settings
import os.path as osp
import pickle

# Load dataset
esnli_dir = osp.join(settings.data_dir, 'esnli')
train_set_1 = pd.read_csv(osp.join(esnli_dir, 'esnli_train_1.csv'))
train_set_2 = pd.read_csv(osp.join(esnli_dir, 'esnli_train_2.csv'))
train_set = pd.concat([train_set_1, train_set_2], ignore_index=True)
train_set = train_set.drop_duplicates()

val_set = pd.read_csv(osp.join(esnli_dir, 'esnli_dev.csv'))
val_set = val_set.drop_duplicates()

test_set = pd.read_csv(osp.join(esnli_dir, 'esnli_test.csv'))
test_set = test_set.drop_duplicates()


# Reduce to toy dataset
def reduce_to_toy(train_set, val_set, test_set):
    train_size = 21000
    val_size = int(1 / 3 * train_size)
    test_size = val_size

    esnli_toy_dir = osp.join(settings.data_dir, 'esnli', 'toy')
    if not os.path.exists(esnli_toy_dir):
        os.mkdir(esnli_toy_dir)

    train_set = train_set.sample(train_size, random_state=0)
    val_set = val_set.sample(val_size, random_state=0)
    test_set = test_set.sample(test_size, random_state=0)

    for split_set in [train_set, val_set, test_set]:
        # Drop useless columns
        useless_columns = [
            'pairID', 'WorkerId',
            'Sentence1_marked_1', 'Sentence2_marked_1',
            'Sentence1_Highlighted_1', 'Sentence2_Highlighted_1',
            'Sentence1_marked_2', 'Sentence2_marked_2',
            'Sentence1_Highlighted_2', 'Sentence2_Highlighted_2',
            'Sentence1_marked_3', 'Sentence2_marked_3',
            'Sentence1_Highlighted_3', 'Sentence2_Highlighted_3'
        ]
        split_set.drop(columns=useless_columns, inplace=True, errors='ignore')

        # # Add sentence1 and sentence2 as a list of words
        # split_set['Sentences'] = split_set['Sentence1'] + '->' + split_set['Sentence2']

    # Save toy dataset
    train_set.to_csv(osp.join(esnli_toy_dir, 'esnli_train.csv'), index=False)
    val_set.to_csv(osp.join(esnli_toy_dir, 'esnli_val.csv'), index=False)
    test_set.to_csv(osp.join(esnli_toy_dir, 'esnli_test.csv'), index=False)


# reduce_to_toy(train_set, val_set, test_set)


# Load toy dataset
esnli_toy_dir = osp.join(settings.data_dir, 'esnli', 'toy')
train_set = pd.read_csv(osp.join(esnli_toy_dir, 'esnli_train.csv'))
val_set = pd.read_csv(osp.join(esnli_toy_dir, 'esnli_val.csv'))
test_set = pd.read_csv(osp.join(esnli_toy_dir, 'esnli_test.csv'))

# Convert categorical variables from String to int representation
split_set_dict  = {'train': train_set, 'val': val_set, 'test': test_set}

for split, split_set in split_set_dict.items():
    ic(f'Starting {split} split')

    split_set['gold_label'] = split_set['gold_label'].astype('category').cat.codes

    split_set = split_set.to_dict(orient='list')

    # # Encode sentences
    # ic(f'Encoding {split} split')
    # split_set['Sentence1_Embeddings'] = bart(split_set['Sentence1'])
    # split_set['Sentence2_Embeddings'] = bart(split_set['Sentence2'])

    # write split_set to pickle_file
    ic(f'Dumping pickle file for {split} split')
    split_set_file = osp.join(esnli_toy_dir, f'esnli_{split}.pkl')
    with open(split_set_file, 'wb') as f:
        pickle.dump(split_set, f)

