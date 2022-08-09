from src.datasets.esnli import get_dataset

splits = {'train', 'val', 'test'}
split_set = {split: get_dataset(split) for split in splits}


def get_sentences(split):
    return set(datapoint['Sentences'] for datapoint in split_set[split])


overlap = set.intersection(*tuple(get_sentences(split) for split in splits))
assert len(overlap) == 0, f'Datasets overlap is {len(overlap)}'
print('Passed !')
