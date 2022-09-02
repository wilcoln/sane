import logging
import math
import os
from typing import List

import torch
from tqdm import tqdm
from transformers import BartTokenizer, BartModel

from src.settings import settings
from src.utils.nn import freeze

logging.basicConfig(level='INFO')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

frozen_bart_tokenizer = BartTokenizer.from_pretrained(f'facebook/bart-{settings.bart_version}')
frozen_bart_model = BartModel.from_pretrained(f'facebook/bart-{settings.bart_version}').to(settings.device)
freeze(frozen_bart_model)


def transformer_mean_pool(token_embeddings, attention_mask=None):
    if attention_mask is None:
        return token_embeddings.mean(dim=1)

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def transformer_cls_pool(token_embeddings, *args, **kwargs):
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
    sentences = [str(sent) for sent in sentences]
    num_sentences = len(sentences)
    batch_size = 128
    batches = (sentences[i:i + batch_size] for i in range(0, num_sentences, batch_size))

    batches = tqdm(batches, total=math.ceil(num_sentences / batch_size)) if verbose else batches

    for i, batch in enumerate(batches):
        encoded_inputs = frozen_bart_tokenizer(batch, max_length=settings.max_length, truncation=True, padding=True,
                                               return_tensors='pt')
        encoded_inputs = {k: v.to(settings.device) for k, v in encoded_inputs.items()}
        model_outputs = frozen_bart_model(**encoded_inputs)
        encoded_batch = transformer_sentence_pool(model_outputs['last_hidden_state'], encoded_inputs['attention_mask'])

        if i == 0:
            encoded_batches = encoded_batch
        else:
            encoded_batches = torch.cat([encoded_batches, encoded_batch], dim=0)

    return encoded_batches.detach()


def tokenize(sentence_list):
    return frozen_bart_tokenizer(sentence_list, max_length=settings.max_length, truncation=True, padding=True,
                                 return_tensors='pt')


def corrupt(tensor, noise_prop=None, sigma2=None):
    if sigma2 is not None:
        return tensor + torch.randn_like(tensor) * sigma2
    return (1 - noise_prop) * tensor + torch.rand_like(tensor) * noise_prop
