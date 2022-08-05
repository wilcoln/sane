import math
import os
import os.path as osp

import pandas as pd
from icecream import ic
from tqdm import tqdm

from src.utils.embeddings import bart
from src.utils.settings import settings
from src.utils.types import ChunkedList
from src.conceptnet import conceptnet as cn


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


def add_concepts(splits, esnli_output_dir):
    ic('Add Knowledge to Data Points')
    for split, split_set in splits.items():
        concepts_path = os.path.join(esnli_output_dir, f'{split}_concept_ids')
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


def preprocess(esnli_frac: float = .01):
    esnli_output_dir = osp.join(settings.data_dir, f'esnli_{settings.data_frac}')
    if not os.path.exists(esnli_output_dir):
        os.mkdir(esnli_output_dir)

    # Read dataset
    ic('Reading dataset')
    splits = read_dataset()
    # Reduce dataset
    ic('Reducing dataset')
    splits = reduce_dataset(splits, esnli_frac)
    # Convert to dict
    ic('Converting to dict')
    splits = df_to_dict(splits)
    # Add concepts
    ic('Adding concepts')
    splits = add_concepts(splits, esnli_output_dir)
    # Encode sentences
    ic('Encoding sentences')
    splits = encode(splits, esnli_output_dir)
    # Save splits
    ic('Saving splits')
    save_splits(splits, esnli_output_dir)
    ic('Done')


if __name__ == 'main':
    preprocess(settings.data_frac)
