import os, random, json, pickle, re
import numpy as np
import torch.utils.data
import pandas as pd
from itertools import chain

class ComVEDataset(torch.utils.data.Dataset):
    """
    A dataset for CQA
    """
    def __init__(self, file_path, preprocess=lambda x: x, test_set=False):
        super().__init__()
        self.preprocess = preprocess
        self.test_set = test_set

        self.comve = pd.read_csv(file_path)

        print('Done.')

    def __len__(self):
        return len(self.comve)

    def __getitem__(self, i):
        row = self.comve.iloc[i]

        # Remove extra annotation on prompt from WP dataset
        premise= self.preprocess(row['a'] + '[SEP]' + row['b'])

        premise_encoded = self.preprocess(premise)
        expl_encoded = self.preprocess(row['explanation'])
        
        return tuple(premise_encoded, row['label'], expl_encoded)


def collate_fn_masked(samples):
    """ Creates a batch out of samples """
    max_len = max(map(len, samples))
    # Zero pad mask
    x_mask = torch.ByteTensor([[1] * len(x) + [0] * (max_len - len(x)) for x in samples])
    x = torch.LongTensor([x + [0] * (max_len - len(x)) for x in samples])
    return x[:, :-1], x[:, 1:].contiguous(), x_mask[:, 1:]