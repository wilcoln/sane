import math
import os
import os.path as osp
import random

import pandas as pd
from icecream import ic
from tqdm import tqdm

from src.conceptnet import conceptnet as cn
from src.settings import settings
from src.utils.embeddings import bart
from src.utils.types import ChunkedList


def read_dataset():
    comve_dir = osp.join(settings.data_dir, 'comve')
    train_set = pd.read_csv(osp.join(comve_dir, 'train.csv'))
    val_set = pd.read_csv(osp.join(comve_dir, 'dev.csv'))
    test_set = pd.read_csv(osp.join(comve_dir, 'test.csv'))

    return {'train': train_set, 'val': val_set, 'test': test_set}


def reduce_dataset(splits, frac):
    for split, split_set in splits.items():
        split_set = split_set.sample(int(len(split_set) * frac), random_state=0)
        # Replace Sentence1 and Sentence2 with Sentences
        sentences = []
        gold_labels = []
        for sentence1, sentence2 in zip(split_set['Correct Statement'], split_set['Incorrect Statement']):
            label = random.randint(0, 1)
            if label:
                sentence = sentence2 + '. ' + sentence1
            else:
                sentence = sentence1 + '. ' + sentence2

            # remove double periods
            sentence = sentence.replace('..', '.')
            sentences.append(sentence)
            gold_labels.append(label)

        split_set['Sentences'] = sentences
        split_set['gold_label'] = gold_labels
        # Drop useless columns
        useless_columns = ['Confusing Reason1', 'Confusing Reason2']
        split_set.drop(columns=useless_columns, inplace=True, errors='ignore')
        # Update split_set
        splits[split] = split_set

    return splits


def encode(splits, output_dir):
    for split, split_set in splits.items():
        sentences_embedding_path = osp.join(output_dir, f'{split}_Sentences_embedding')
        try:
            # Load Sentences_embeddings
            ic('Loading Sentences_embeddings')
            split_set['Sentences_embedding'] = ChunkedList(n=len(split_set['Sentences']),
                                                           dirpath=sentences_embedding_path)
        except FileNotFoundError:
            ic('Computing Sentences_embeddings')
            # Create sentence embeddings
            split_set['Sentences_embedding'] = ChunkedList(lst=split_set['Sentences'], num_chunks=math.ceil(
                len(split_set['Sentences']) / settings.chunk_size)).apply(lambda l: bart(l, verbose=True),
                                                                          dirpath=sentences_embedding_path)
        # Update split_set
        splits[split] = split_set
    return splits


def df_to_dict(splits):
    return {split: split_set.to_dict(orient='list') for split, split_set in splits.items()}


def save_splits(splits, output_dir):
    for split, split_set in splits.items():
        for k, v in split_set.items():
            if not isinstance(v, ChunkedList):
                k_path = osp.join(output_dir, f'{split}_{k}')
                ChunkedList(lst=v, num_chunks=math.ceil(len(split_set['Sentences']) / settings.chunk_size)).to_big(
                    k_path)


def compute_concept_ids(sentence_list):
    return [
        cn.nodes2ids(cn.subgraph(cn.search(sentence), radius=1).nodes)
        for sentence in tqdm(sentence_list)
    ]


def add_concepts(splits, comve_output_dir):
    ic('Add Knowledge to Data Points')
    for split, split_set in splits.items():
        concepts_path = os.path.join(comve_output_dir, f'{split}_concept_ids')
        try:
            ic(f'Loading Concept ids for {split} split')
            split_set['concept_ids'] = ChunkedList(n=len(split_set['Sentences']), dirpath=concepts_path)
        except FileNotFoundError:
            ic(f'Computing concept ids for {split} split')
            split_set['concept_ids'] = ChunkedList(lst=split_set['Sentences'], num_chunks=math.ceil(
                len(split_set['Sentences']) / settings.chunk_size)).apply(
                lambda l: compute_concept_ids(l), concepts_path)
        # Update split_set
        splits[split] = split_set

    return splits


def preprocess(comve_frac: float = .01):
    comve_output_dir = osp.join(settings.data_dir, f'comve_{settings.data_frac}')
    if not os.path.exists(comve_output_dir):
        os.mkdir(comve_output_dir)

    # Read dataset
    ic('Reading dataset')
    splits = read_dataset()
    # Reduce dataset
    ic('Reducing dataset')
    splits = reduce_dataset(splits, comve_frac)
    # Convert to dict
    ic('Converting to dict')
    splits = df_to_dict(splits)
    # Add concepts
    ic('Adding concepts')
    splits = add_concepts(splits, comve_output_dir)
    # Encode sentences
    ic('Encoding sentences')
    splits = encode(splits, comve_output_dir)
    # Save splits
    ic('Saving splits')
    save_splits(splits, comve_output_dir)
    ic('Done')


if __name__ == 'main':
    preprocess(settings.data_frac)
