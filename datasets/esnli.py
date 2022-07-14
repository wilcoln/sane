import torch.utils.data
import pandas as pd
import os.path as osp
from tqdm import tqdm
from utils.settings import settings
import pickle
from icecream import ic


class ESNLIDataset(torch.utils.data.Dataset):
    """
    A dataset for ESNLI
    """

    def __init__(self, path: str, split: str = 'train', frac=1.0):
        assert split in {'train', 'val', 'test'}, 'split must be one of train, val, test'
        assert 0.0 <= frac <= 1.0, 'frac must be between 0 and 1'

        super().__init__()
        self.name = f'esnli_{split}'

        # Load pickle file
        esnli_path = osp.join(path, f'{self.name}.pkl')
        self.esnli = pickle.load(open(esnli_path, 'rb'))

    def __len__(self):
        return len(self.esnli['gold_label'])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.esnli.items()}

    def add_conceptnet(self, save=True):
        conceptnet_path = osp.join(settings.data_dir, 'conceptnet', 'conceptnet-assertions-5.6.0_cleaned.csv')
        cn = pd.read_csv(conceptnet_path)

        # tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        # encoded_inputs = tokenizer(self.esnli['Sentence1'].tolist(), max_length=1024, truncation=True, padding=True)
        # encoded_labels = tokenizer(self.esnli['Sentence2'].tolist(), max_length=1024, truncation=True, padding=True)
        #
        # # get tokens as words from encoded inputs
        # inputs = [tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False) for tokens in
        #           encoded_inputs['input_ids']]
        # # get tokens from encoded labels
        # labels = [tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False) for tokens in
        #           encoded_labels['input_ids']]
        #
        # # For each row in esnli, get the triples in cn with matching inputs and labels
        # ic(len(inputs), len(labels), len(self.esnli))
        #
        # self.esnli.reset_index(drop=True, inplace=True)

        # for i, row in self.esnli.iterrows():
        #     ic(i, row)
        #     # get the triples in cn with matching inputs and labels
        #     triples = cn[cn['source'].isin(ic(inputs[i])) | cn['target'].isin(labels[i])]
        #     # get the triples in cn with matching inputs and labels and add them to the esnli dataframe
        #     self.esnli.loc[i, 'triples'] = triples.to_dict('records')

        self.esnli.reset_index(drop=True, inplace=True)
        for i, row in tqdm(self.esnli.iterrows(), total=len(self.esnli)):
            row_vocab = set(row['Sentences'].lower().split(' '))
            # get the triples in cn with matching inputs and labels
            triples = cn[cn['source'].isin(row_vocab) | cn['target'].isin(row_vocab)].index.astype('str').tolist()
            # get the triples in cn with matching inputs and labels and add them to the esnli dataframe
            self.esnli.loc[i, 'Triple_ids'] = ', '.join(triples)

        # Drop useless columns
        self.esnli.drop(columns=['Sentences'], inplace=True, errors='ignore')
        # Save the augmented dataframe to a csv file

        if save:
            self.esnli.to_csv(osp.join(settings.data_dir, 'esnli', f'{self.name}_conceptnet.csv'), index=False)
