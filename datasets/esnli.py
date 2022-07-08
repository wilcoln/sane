import torch.utils.data
import pandas as pd
import os.path as osp


class ESNLIDataset(torch.utils.data.Dataset):
    """
    A dataset for ESNLI
    """
    def __init__(self, path: str, split: str = 'train', frac=1.0):
        assert split in {'train', 'val', 'test'}, 'split must be one of train, val, test'
        assert 0.0 <= frac <= 1.0, 'frac must be between 0 and 1'

        super().__init__()
        self.name = 'esnli'

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

    def __len__(self):
        return len(self.esnli)

    def __getitem__(self, i):
        return dict(self.esnli.iloc[i])
