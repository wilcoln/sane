import os
import os.path as osp
from typing import List
import pickle

import logging
from tqdm import tqdm
import numpy as np
import torch
from transformers import BartTokenizer, BartModel

from utils.settings import settings
from sentence_transformers import SentenceTransformer
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
import math

logging.basicConfig(level='INFO')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model = BartModel.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


def transformer_mean_pooling(model_outputs, encoded_inputs):
    token_embeddings = model_outputs['encoder_last_hidden_state']
    input_mask_expanded = encoded_inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def bart(sentences: List[str], verbose: bool = False) -> torch.Tensor:
    """Compute bart embeddings for texts.
    Args:
        sentences: text to use for each node
    Returns:
        embeddings: embeddings for each node

    """

    # ic | encoded_input.keys(): dict_keys(['input_ids', 'attention_mask'])
    # ic | model_output.keys(): odict_keys(['last_hidden_state', 'past_key_values', 'encoder_last_hidden_state'])
    num_sentences = len(sentences)
    batch_size = 256
    batches = (sentences[i:i + batch_size] for i in range(0, num_sentences, batch_size))

    batches = tqdm(batches, total=math.ceil(num_sentences/batch_size)) if verbose else batches

    for i, batch in enumerate(batches):
        encoded_inputs = tokenizer(batch, max_length=512, truncation=True, padding=True, return_tensors='pt')
        model_outputs = model(**encoded_inputs)
        encoded_batch = transformer_mean_pooling(model_outputs, encoded_inputs)

        if i == 0:
            encoded_batches = encoded_batch
        else:
            encoded_batches = torch.cat([encoded_batches, encoded_batch], dim=0)

    return encoded_batches.detach()


def sbert(sentences: List[str], verbose: bool = False) -> torch.Tensor:
    """Compute sbert embeddings for nodes of a knowledge graph
    Args:
        sentences: text to use for each node
        verbose: Whether to show progress bar
    Returns:
        A map nodes and their embeddings
    """

    if not verbose:
        # Disable logger
        from sentence_transformers.SentenceTransformer import logger
        logger.disabled = True

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Compute embedding for both lists
    return model.encode(sentences, convert_to_tensor=True, batch_size=settings.batch_size, device=str(
        settings.device), show_progress_bar=verbose)


def fasttext(texts: List[str]) -> List[np.array]:

    """Compute fasttext embeddings for texts.
    Args:
        texts: text to use for each node
    Returns:
        A map nodes and their embeddings
    """

    # initialize the word embeddings
    word_embedding = WordEmbeddings('en')

    # initialize the document embeddings, mode = mean
    document_embeddings = DocumentPoolEmbeddings([word_embedding])

    # sentences

    sentences = [Sentence(text) for text in texts]

    for sentence in sentences:
        document_embeddings.embed(sentence)

    # Compute embeddings
    embeddings = [sentence.embedding for sentence in sentences]

    return embeddings
