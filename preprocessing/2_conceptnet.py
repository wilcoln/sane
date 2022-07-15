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
import nltk


def remove_qualifier(string):
    string = str(string)
    return string.split('/')[0]

def tokenize(sentence):
    sentence = str(sentence)
    stop = set(stopwords.words('english') + list(string.punctuation))
    tokens = '|'.join(set(i for i in word_tokenize(sentence.replace('_', ' ').lower()) if i not in stop))
    return tokens

esnli_toy_dir = osp.join(settings.data_dir, 'esnli', 'toy')
# conceptnet_path = osp.join(settings.data_dir, 'conceptnet', 'conceptnet-assertions-5.6.0_cleaned.csv')
# cn_vocab_path = conceptnet_path[:-4] + '_vocab.csv'

# try:
#     ic('Loading Conceptnet Vocab')
#     cn = pd.read_csv(cn_vocab_path)
#     cn = cn.sample(frac=.01, random_state=0)
# except:
#     ic('Creating Conceptnet vocab')
#     conceptnet_path = osp.join(settings.data_dir, 'conceptnet', 'conceptnet-assertions-5.6.0_cleaned.csv')
#     cn = pd.read_csv(conceptnet_path)
#     cn = cn.sample(frac=.01, random_state=0)

#     cn = cn[['source', 'target']]

#     cn['Vocab'] = (cn['source'].apply(remove_qualifier) + ' ' + cn['target'].apply(remove_qualifier)).apply(tokenize)

#     cn = cn[['Vocab']]

#     cn.to_csv(cn_vocab_path)


# Load toy dataset
splits = ['train', 'val', 'test']
for split in splits:
    # esnli_path = osp.join(esnli_toy_dir, f'esnli_{split}.pkl')
    # esnli = pickle.load(open(esnli_path, 'rb'))

    # sentence_1_list = esnli['Sentence1']
    # sentence_2_list = esnli['Sentence2']

    # esnli['Vocab'] = [
    #     tokenize(sentence1 + ' ' + sentence2)
    #     for sentence1, sentence2 in zip(sentence_1_list, sentence_2_list)
    # ]

    # esnli = {k: v for k, v in esnli.items() if k == 'Vocab'}

    # esnli['Triple_ids'] = []

    # for row_vocab in tqdm(esnli['Vocab']):
    #     # get the triples in cn with matching inputs and labels
    #     filt = cn['Vocab'].str.contains(row_vocab, na=False)
    #     triple_ids = cn[filt].index.tolist()
    #     esnli['Triple_ids'].append(triple_ids)

    # ic('Dumping')
    # esnli_cn_path = osp.join(esnli_toy_dir, f'esnli_cn_{split}.pkl')
    # with open(esnli_cn_path, 'wb') as f:
    #     pickle.dump(esnli, f)

#######################
    # Compute pyg Data
    esnli_cn_path = osp.join(esnli_toy_dir, f'esnli_cn_{split}.pkl')
    esnli_cn = pickle.load(open(esnli_cn_path, 'rb'))

    esnli_pyg = triple_ids_to_pyg_data(esnli_cn['Triple_ids'])
    ic('Dumping')
    esnli_pyg_path = osp.join(esnli_toy_dir, f'esnli_pyg_{split}.pkl')
    with open(esnli_pyg_path, 'wb') as f:
        pickle.dump(esnli_pyg, f)
#########################"

    # # Add pyg_data
    # esnli_toy_dir = osp.join(settings.data_dir, 'esnli', 'toy')


    # esnli_path = osp.join(esnli_toy_dir, f'esnli_{split}.pkl')
    # esnli = pickle.load(open(esnli_path, 'rb'))

    # esnli_pyg_path = osp.join(esnli_toy_dir, f'esnli_pyg_{split}.pkl')
    # esnli_pyg = pickle.load(open(esnli_pyg_path, 'rb'))

    # esnli['pyg_data'] = esnli_pyg


    # with open(esnli_path, 'wb') as f:
    #     pickle.dump(esnli, f)

#############################################""
