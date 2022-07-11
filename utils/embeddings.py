import os
from typing import List

import logging
import numpy as np
import torch

from utils.settings import settings
from sentence_transformers import SentenceTransformer
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence

logging.basicConfig(level='INFO')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def bert(sentences: List[str], verbose: bool = False) -> torch.Tensor:
    """Compute bert embeddings for nodes of a knowledge graph
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
