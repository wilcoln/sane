from typing import List

from nltk.util import everygrams


def all_grams(sentence) -> List[str]:
    return set('_'.join(ngram) for ngram in everygrams(sentence.split(), min_len=1, max_len=len(sentence.split(' '))))
