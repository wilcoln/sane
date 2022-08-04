import argparse
import os.path as osp

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', help='Number of epochs', type=int, default=5)
parser.add_argument('--batch_size', help='Batch size', type=int, default=128)
parser.add_argument('--sent_dim', help='sent_dim', type=int, default=768)  # Bart-base d_model = 768
parser.add_argument('--chunk_size', help='Chunk size', type=int, default=10000)
parser.add_argument('--colab', action='store_true', help="whether we are running on google colab", default=False)
parser.add_argument('--dataset', help='Dataset to use', type=str)
parser.add_argument('--ignore_datasets', nargs='*', help='Datasets to ignore', type=str)
_data_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
parser.add_argument('--data_dir', help='Data dir', type=str, default=_data_dir)
_cache_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'cache')
parser.add_argument('--cache_dir', help='Data dir', type=str, default=_cache_dir)
_results_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'results')
parser.add_argument('--results_dir', help='Result directory to use', type=str, default=_results_dir)
parser.add_argument('--no_show', action='store_true', help="Do not show the figure at the end", default=False)
parser.add_argument('--persistent_workers', action='store_true', help="Whether to make dataloader workers "
                                                                      "persistent", default=False)
parser.add_argument('--num_workers', help='Number of workers', type=int, default=0)
parser.add_argument('--hidden_dim', help='Hidden Dimension', type=int, default=32)
parser.add_argument('--lstm_num_inputs', help='Number of inputs for lstm aggregator', type=int)
parser.add_argument('--no_eval_train', action='store_true', help="whether to evaluate on the train set as well",
                    default=False)
parser.add_argument('--no_extensions', action='store_true', help="whether to include extension in the experiments",
                    default=False)
parser.add_argument('--num_runs', help='Number of runs', type=int, default=1)
parser.add_argument('--std', action='store_true', help='Include standard deviation in table output', default=False)
parser.add_argument('--alpha', help='Alpha', type=float, default=.4)
parser.add_argument('--data_frac', help='Fraction of data to use', type=float, default=.25)
settings, unknown = parser.parse_known_args()
setattr(settings, 'device', torch.device('cuda'))

if settings.colab:
    settings.data_dir = '/content/data'
    settings.results_dir = '/content/results'
    settings.num_workers = 2
