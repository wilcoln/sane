import pandas as pd

import os
import os.path as osp
import pickle
import string

from icecream import ic
from tqdm import tqdm

from utils.graphs import triple_ids_to_pyg_data
from utils.settings import settings
from utils.embeddings import bart
from nltk import word_tokenize
from nltk.corpus import stopwords


def _read_dataset():
    esnli_dir = osp.join(settings.data_dir, 'esnli')
    train_set_1 = pd.read_csv(osp.join(esnli_dir, 'esnli_train_1.csv'))
    train_set_2 = pd.read_csv(osp.join(esnli_dir, 'esnli_train_2.csv'))
    train_set = pd.concat([train_set_1, train_set_2], ignore_index=True)
    train_set = train_set.drop_duplicates()

    val_set = pd.read_csv(osp.join(esnli_dir, 'esnli_dev.csv'))
    val_set = val_set.drop_duplicates()

    test_set = pd.read_csv(osp.join(esnli_dir, 'esnli_test.csv'))
    test_set = test_set.drop_duplicates()

    return {'train': train_set, 'val': val_set, 'test': test_set}


def _reduce_dataset(splits, frac):
    for split, split_set in splits.items():
        split_set = split_set.sample(int(len(split_set)*frac), random_state=0)
        # Replace Sentence1 and Sentence2 with Sentences
        split_set['Sentences'] = split_set['Sentence1'] + '. ' + split_set['Sentence2']
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
        # Add vocabulary
        split_set['Vocab'] = split_set['Sentences'].apply(_tokenize)

        # Update split_set
        splits[split] = split_set

    return splits


def _encode(splits):
    for split, split_set in splits.items():
        split_set['Sentences_embedding'] = bart(split_set['Sentences']).detach()
        # Update split_set
        splits[split] = split_set
    return splits


def _df_to_dict(splits):
    return {split: split_set.to_dict(orient='list') for split, split_set in splits.items()}


def _save_splits(splits, output_dir):
    esnli_frac_dir = osp.join(settings.data_dir, 'esnli', output_dir)
    if not os.path.exists(esnli_frac_dir):
        os.mkdir(esnli_frac_dir)
    for split, split_set in splits.items():
        ic(f'Dumping pickle file for {split} split')
        split_set_file = osp.join(esnli_frac_dir, f'esnli_{split}.pkl')
        with open(split_set_file, 'wb') as f:
            pickle.dump(split_set, f)


def _remove_qualifier(string):
    string = str(string)
    return string.split('/')[0]


def _tokenize(sentence):
    sentence = str(sentence)
    stop = set(stopwords.words('english') + list(string.punctuation))
    tokens = '|'.join(set(i for i in word_tokenize(sentence.replace('_', ' ').lower()) if i not in stop))
    return tokens


def _add_concepts(splits, conceptnet_frac):
    cn_path = osp.join(settings.data_dir, 'conceptnet', 'conceptnet-assertions-5.6.0_cleaned.csv')
    # Creating vocabulary for conceptnet
    ic('Creating Conceptnet vocab')
    cn = pd.read_csv(cn_path)
    cn = cn.sample(frac=conceptnet_frac, random_state=0)
    cn = cn[['source', 'target']]
    cn['Vocab'] = (cn['source'].apply(_remove_qualifier) + ' ' + cn['target'].apply(_remove_qualifier)).apply(_tokenize)

    for split, split_set in splits.items():
        split_set['Triple_ids'] = []

        # Add triple ids to the dataset
        for row_vocab in tqdm(split_set['Vocab']):
            # get the triples in cn with matching inputs and labels
            triple_ids = cn[cn['Vocab'].str.contains(row_vocab, na=False)].index.tolist()
            split_set['Triple_ids'].append(triple_ids)

        # Compute pyg Data
        ic('Computing Pyg Data')
        split_set['pyg_data'] = triple_ids_to_pyg_data(split_set['Triple_ids'])

        # Update split_set
        splits[split] = split_set

    return splits


def preprocess(esnli_frac: float = .01, conceptnet_frac: float = .01, output_dir: str = None):
    # Read dataset
    ic('Reading dataset')
    splits = _read_dataset()
    # Reduce dataset
    ic('Reducing dataset')
    splits = _reduce_dataset(splits, esnli_frac)
    # Convert to dict
    ic('Converting to dict')
    splits = _df_to_dict(splits)
    # Add concepts
    ic('Adding concepts')
    splits = _add_concepts(splits, conceptnet_frac)
    # Encode sentences
    ic('Encoding sentences')
    splits = _encode(splits)
    # Save splits
    ic('Saving splits')
    _save_splits(splits, output_dir)
    ic('Done')


preprocess(settings.data_frac, settings.data_frac, f'esnli_frac_{settings.data_frac}')

