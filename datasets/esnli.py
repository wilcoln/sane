import torch.utils.data
import pandas as pd
import os.path as osp

from utils.settings import settings


class ESNLIDataset(torch.utils.data.Dataset):
    """
    A dataset for ESNLI
    """
    def __init__(self, path: str, split: str = 'train', frac=1.0, add_conceptnet=False):
        assert split in {'train', 'val', 'test'}, 'split must be one of train, val, test'
        assert 0.0 <= frac <= 1.0, 'frac must be between 0 and 1'

        super().__init__()
        self.name = f'esnli_{split}'

        if split == 'train':
            train_set_1 = pd.read_csv(osp.join(path, 'esnli_train_1.csv'))
            train_set_2 = pd.read_csv(osp.join(path, 'esnli_train_2.csv'))
            self.esnli = pd.concat([train_set_1, train_set_2], axis=0, ignore_index=True)
        elif split == 'val':
            self.esnli = pd.read_csv(osp.join(path, 'esnli_dev.csv'))
        elif split == 'test':
            self.esnli = pd.read_csv(osp.join(path, 'esnli_test.csv'))

        # Randomly sample a subset of the data
        if frac < 1:
            self.esnli = self.esnli.sample(frac=frac)

        # Drop useless columns
        self.esnli.drop(columns=['pairID', 'WorkerId'], inplace=True, errors='ignore')

        # Convert categorical variables from String to int representation
        self.esnli['gold_label'] = self.esnli['gold_label'].astype('category').cat.codes

        # Add sentence1 and sentence2 as a list of words
        self.esnli['Sentences'] = self.esnli['Sentence1'] + '->' + self.esnli['Sentence2']

        # Add conceptnet data
        if add_conceptnet is not None:
            self.augment()

    def __len__(self):
        return len(self.esnli)

    def __getitem__(self, i):
        return dict(self.esnli.iloc[i])

    def augment(self):
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

        for i, row in self.esnli.iterrows():
            row_vocab = set(row['Sentences'].split(' '))
            # get the triples in cn with matching inputs and labels
            triples = cn[cn['source'].isin(row_vocab) | cn['target'].isin(row_vocab)].index.astype('str').tolist()
            # get the triples in cn with matching inputs and labels and add them to the esnli dataframe
            self.esnli.loc[i, 'triple_ids'] = ', '.join(triples)

        # Save the augmented dataframe to a csv file
        self.esnli.to_csv(osp.join(settings.data_dir, 'esnli', f'{self.name}_augmented.csv'), index=False)









