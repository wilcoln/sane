import pickle

import pandas as pd
import os.path as osp
import json

from icecream import ic
from utils.settings import settings


esnli_toy_dir = osp.join(settings.data_dir, 'esnli', 'toy')
esnli = pickle.load(open(osp.join(esnli_toy_dir, 'esnli_train.pkl'), 'rb'))
sentence_1_list = esnli['Sentence1']
sentence_2_list = esnli['Sentence2']

sentence_list = sentence_1_list + sentence_2_list

vocabulary = set(word for sentence in sentence_list for word in sentence.split())
ic(len(vocabulary))



# Reduce size of conceptnet by concepts that are not in toy esnli dataset