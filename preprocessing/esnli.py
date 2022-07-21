import pandas as pd

import os
import os.path as osp
import pickle
import string

from icecream import ic
from tqdm import tqdm
import math

from utils.graphs import concept_ids_to_pyg_data, concept_df
from utils.settings import settings
from utils.embeddings import bart
from nltk import word_tokenize
from nltk.corpus import stopwords
from utils.types import ChunkedList
import shutil

def _read_dataset():
    esnli_dir = osp.join(settings.data_dir, 'esnli')
    train_set_1 = pd.read_csv(osp.join(esnli_dir, 'esnli_train_1.csv'))
    train_set_2 = pd.read_csv(osp.join(esnli_dir, 'esnli_train_2.csv'))
    train_set = pd.concat([train_set_1, train_set_2], ignore_index=True)

    val_set = pd.read_csv(osp.join(esnli_dir, 'esnli_dev.csv'))

    test_set = pd.read_csv(osp.join(esnli_dir, 'esnli_test.csv'))

    return {'train': train_set, 'val': val_set, 'test': test_set}


def _reduce_dataset(splits, frac, output_dir):
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
        vocab_path = osp.join(output_dir, f'{split}_vocab.pkl')
        try:
            # Load esnli vocab
            ic('Load esnli Vocab')
            split_set['Vocab'] = pickle.load(open(os.path.join(vocab_path), 'rb'))
        except:
            ic('Create vocabulary for split')
            split_set['Vocab'] = split_set['Sentences'].apply(_tokenize)
            with open(vocab_path, 'wb') as f:
                pickle.dump(split_set['Vocab'], f)

        # Update split_set
        splits[split] = split_set

    return splits


def _encode(splits, output_dir):
    for split, split_set in splits.items():
        sentences_embedding_path = osp.join(output_dir, f'{split}_Sentences_embedding')
        try:
            # Load Sentences_embeddings
            split_set['Sentences_embedding'] = ChunkedList(n=len(split_set['Sentences']), dirpath=sentences_embedding_path)
        except:
            # Create sentence embeddings
            split_set['Sentences_embedding'] = ChunkedList(lst=split_set['Sentences'], num_chunks=math.ceil(len(split_set['Sentences'])/settings.chunk_size)).apply(lambda l : bart(l), dirpath=sentences_embedding_path)
        # Update split_set
        splits[split] = split_set
    return splits


def _df_to_dict(splits):
    return {split: split_set.to_dict(orient='list') for split, split_set in splits.items()}


def _save_splits(splits, output_dir):
    for split, split_set in splits.items():
        for k, v in split_set.items():
            if not isinstance(v, ChunkedList):
                k_path = osp.join(output_dir, f'{split}_{k}')
                ChunkedList(lst=v, num_chunks=math.ceil(len(split_set['Sentences'])/settings.chunk_size)).to_big(k_path)

def _remove_qualifier(string):
    string = str(string)
    return string.split('/')[0]


def _tokenize(sentence):
    sentence = str(sentence)
    stop = set(stopwords.words('english') + list(string.punctuation))
    tokens = '|'.join(set(i for i in word_tokenize(sentence.replace('_', ' ').lower()) if i not in stop))
    return tokens

def _compute_concept_ids(cn, vocab_list):
    concept_ids_list = []
    for row_vocab in tqdm(vocab_list):
        # get the triples in cn with matching inputs and labels
        concept_ids = cn[cn['Vocab'].str.contains(row_vocab, na=False)].index.tolist() # TODO: Semantic similarity
        concept_ids_list.append(concept_ids)
    return concept_ids_list

def _add_concepts(splits, esnli_output_dir):
    cn = concept_df
    # Add Conceptnet Vocab
    output_dir = osp.join(settings.data_dir, f'conceptnet')
    vocab_path = osp.join(output_dir, 'vocab.pkl')
    try:
        # Load conceptnet vocab
        ic('Load conceptnet Vocab')
        cn['Vocab'] = pickle.load(open(os.path.join(vocab_path), 'rb'))
    except:
        # Creating vocabulary for conceptnet
        ic('Creating Conceptnet vocab')
        cn['Vocab'] = (cn['name'].apply(_remove_qualifier)).apply(_tokenize)

        with open(vocab_path, 'wb') as f:
            pickle.dump(cn['Vocab'], f)
    
    cn = cn.sample(frac=.1, random_state=0) #  TODO: Do Something about it
    ic('Add Knowledge to Data Points')
    for split, split_set in splits.items():
        concepts_path = os.path.join(esnli_output_dir, f'{split}_concept_ids')
        try:
            ic(f'Loading Concept ids for {split} split')
            split_set['concept_ids'] = ChunkedList(n=len(split_set['Sentences']), dirpath=concepts_path)
        except:
            ic(f'Computing concept ids for {split} split')
            split_set['concept_ids'] = ChunkedList(lst=split_set['Vocab'], num_chunks=math.ceil(len(split_set['Sentences'])/settings.chunk_size)).apply(lambda l: _compute_concept_ids(cn, l), concepts_path)

        # Delete vocab for split
        shutil.rmtree(os.path.join(esnli_output_dir, f'{split}_vocab'), ignore_errors=True)

        # Adding Pyg Data
        pyg_data_path = os.path.join(esnli_output_dir, f'{split}_pyg_data')
        try:
            ic(f'Loading Pyg Data for {split} split')
            split_set['pyg_data'] = ChunkedList(n=len(split_set['Sentences']), dirpath=pyg_data_path)
        except:
        # Compute pyg Data
            ic(f'Computing Pyg Data for {split} split')
            split_set['pyg_data'] = split_set['concept_ids'].apply(concept_ids_to_pyg_data, pyg_data_path)

        # Delete concept ids
        shutil.rmtree(os.path.join(esnli_output_dir, f'{split}_concept_ids'), ignore_errors=True)
        

        # Remove Vocab
        del split_set['Vocab']
        del split_set['concept_ids']

        # Update split_set
        splits[split] = split_set

    return splits


def preprocess(esnli_frac: float = .01):
    esnli_output_dir = osp.join(settings.data_dir, f'esnli_{settings.data_frac}')
    if not os.path.exists(esnli_output_dir):
        os.mkdir(esnli_output_dir)

    # Read dataset
    ic('Reading dataset')
    splits = _read_dataset()
    # Reduce dataset
    ic('Reducing dataset')
    splits = _reduce_dataset(splits, esnli_frac, esnli_output_dir)
    # Convert to dict
    ic('Converting to dict')
    splits = _df_to_dict(splits)
    # # Add concepts
    ic('Adding concepts')
    splits = _add_concepts(splits, esnli_output_dir)
    # Encode sentences
    ic('Encoding sentences')
    splits = _encode(splits, esnli_output_dir)
    # Save splits
    ic('Saving splits')
    _save_splits(splits, esnli_output_dir)
    ic('Done')



preprocess(settings.data_frac)
