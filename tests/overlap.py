from src.datasets.nl import get_dataset
from src.settings import settings


def test_overlap(name):
    splits = {'train', 'val', 'test'}
    split_set = {split: get_dataset(split, name) for split in splits}

    def get_sentences(split):
        return set(datapoint['Sentences'] for datapoint in split_set[split])

    sets = tuple(get_sentences(split) for split in splits)

    print(f'Set lengths : {[len(s) for s in sets]}')
    overlap = set.intersection(*sets)
    assert len(overlap) == 0, f'Datasets overlap is {len(overlap)}'
    print('Passed !')


if __name__ == '__main__':
    test_overlap(settings.dataset)
