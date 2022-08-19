from src.datasets.esnli import get_loader as esnli_get_loader


def get_loader(split, dataset=None):
    return esnli_get_loader(split, dataset, 'comve')
