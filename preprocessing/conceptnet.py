import pickle
import string

import pandas as pd
import os.path as osp
import json

from icecream import ic
from tqdm import tqdm

from utils.graphs import triple_ids_to_pyg_data
from utils.settings import settings
from nltk import word_tokenize
from nltk.corpus import stopwords

# # Load toy dataset
# esnli_toy_dir = osp.join(settings.data_dir, 'esnli', 'toy')
# esnli = pickle.load(open(osp.join(esnli_toy_dir, 'esnli_train.pkl'), 'rb'))


def tokenize(sentence):
    sentence = str(sentence)
    stop = set(stopwords.words('english') + list(string.punctuation) + ['/'])
    tokens = set(i for i in word_tokenize(sentence.replace('/', ' ').lower()) if i not in stop)
    return tokens


esnli_toy_dir = osp.join(settings.data_dir, 'esnli', 'toy')
esnli = pd.read_csv(osp.join(esnli_toy_dir, 'esnli_train.csv'))
esnli['gold_label'] = esnli['gold_label'].astype('category').cat.codes
esnli = esnli.to_dict(orient='list')

sentence_1_list = esnli['Sentence1']
sentence_2_list = esnli['Sentence2']

esnli['Vocab'] = [
    tokenize(sentence1 + ' ' + sentence2)
    for sentence1, sentence2 in zip(sentence_1_list, sentence_2_list)
]


conceptnet_path = osp.join(settings.data_dir, 'conceptnet', 'conceptnet-assertions-5.6.0_cleaned.csv')
cn = pd.read_csv(conceptnet_path)

cn['Vocab'] = (cn['source'] + ' ' + cn['target']).apply(tokenize)

esnli['Triple_ids'] = []

for row_vocab in tqdm(esnli['Vocab']):
    # get the triples in cn with matching inputs and labels
    esnli['Triple_ids'].append(cn[cn['source'].contains('|'.join(row_vocab)) | cn['target'].contains('|'.join(
        row_vocab))].index.tolist())


ic(esnli)
