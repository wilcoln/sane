import logging
import math
import os
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import BartTokenizer, BartModel

from src.settings import settings

logging.basicConfig(level='INFO')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model = BartModel.from_pretrained("facebook/bart-base")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


def transformer_mean_pool(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def transformer_cls_pool(token_embeddings):
    return token_embeddings[:, 0, :]


TRANSFORMER_SENTENCE_POOL = {
    'mean': transformer_mean_pool,
    'cls': transformer_cls_pool
}

transformer_sentence_pool = TRANSFORMER_SENTENCE_POOL[settings.sentence_pool]


def bart(sentences: List[str], verbose: bool = False) -> torch.Tensor:
    """Compute bart embeddings for texts.
    Args:
        sentences: text to use for each node
        verbose: Whether to show progress bar
    Returns:
        embeddings: embeddings for each node

    """

    # ic | encoded_input.keys(): dict_keys(['input_ids', 'attention_mask'])
    # ic | model_output.keys(): odict_keys(['last_hidden_state', 'past_key_values', 'encoder_last_hidden_state'])
    sentences = [str(sent) for sent in sentences]
    num_sentences = len(sentences)
    batch_size = 128
    batches = (sentences[i:i + batch_size] for i in range(0, num_sentences, batch_size))

    batches = tqdm(batches, total=math.ceil(num_sentences / batch_size)) if verbose else batches

    for i, batch in enumerate(batches):
        encoded_inputs = tokenizer(batch, max_length=512, truncation=True, padding=True, return_tensors='pt')
        model_outputs = model(**encoded_inputs)
        encoded_batch = transformer_sentence_pool(model_outputs['last_hidden_state'], encoded_inputs['attention_mask'])

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


def tokenize(sentence_list):
    return tokenizer(sentence_list, max_length=512, truncation=True, padding=True, return_tensors='pt')
