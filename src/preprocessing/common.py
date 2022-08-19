import math
import os
import os.path as osp
from icecream import ic
from tqdm import tqdm

from src.settings import settings
from src.conceptnet import conceptnet as cn
from src.utils.embeddings import bart
from src.utils.types import ChunkedList


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


def add_concepts(splits, dataset_output_dir):
    ic('Add Knowledge to Data Points')
    for split, split_set in splits.items():
        concepts_path = os.path.join(dataset_output_dir, f'{split}_concept_ids')
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


def preprocess(name, read_dataset, reduce_dataset, data_frac: float = .01):
    dataset = osp.join(settings.data_dir, f'{name}_{settings.data_frac}')
    if not os.path.exists(dataset):
        os.mkdir(dataset)

    # Read dataset
    ic('Reading dataset')
    splits = read_dataset()
    # Reduce dataset
    ic('Reducing dataset')
    splits = reduce_dataset(splits, data_frac)
    # Convert to dict
    ic('Converting to dict')
    splits = df_to_dict(splits)
    # Add concepts
    ic('Adding concepts')
    splits = add_concepts(splits, dataset)
    # Encode sentences
    ic('Encoding sentences')
    splits = encode(splits, dataset)
    # Save splits
    ic('Saving splits')
    save_splits(splits, dataset)
    ic('Done')
