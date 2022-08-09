import argparse
import os.path as osp

import torch

parser = argparse.ArgumentParser()

# Experiment settings
exp_settings = [
    ('exp_id', 'Experiment ID', str, None),
    ('exp_desc', 'Experiment description', str, None),
    ('num_epochs', 'Number of epochs', int, 5),
    ('batch_size', 'Batch size', int, 128),
    ('lr', 'Learning rate', float, 1e-4),
    ('sent_dim', 'Sentence dimension', int, 768),  # Bart-base d_model = 768
    ('hidden_dim', 'Hidden dimension', int, 64),
    ('nle_pred', 'Predict with NLE only', bool, False),
    ('max_concepts_per_sent', 'Max concepts per sentence', int, 200),
    ('sentence_pool', 'Sentence pool', str, 'mean'),
    ('data_frac', 'Data fraction', float, 1.0),
    ('alpha', 'NLE loss weight in total loss', float, 0.4),
    ('num_attn_heads', 'Number of heads of the knowledge attention', int, 1),

]
for name, desc, type_, default in exp_settings:
    if type_ is bool:
        parser.add_argument(f'--{name}', action='store_true', help=desc, default=default)
    else:
        parser.add_argument(f'--{name}', type=type_, help=desc, default=default)

# General settings
parser.add_argument('--chunk_size', help='Chunk size', type=int, default=10000)
_data_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
parser.add_argument('--data_dir', help='Data dir', type=str, default=_data_dir)
_cache_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'cache')
parser.add_argument('--cache_dir', help='Data dir', type=str, default=_cache_dir)
_results_dir = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'results')
parser.add_argument('--results_dir', help='Result directory', type=str, default=_results_dir)
parser.add_argument('--input_dir', help='Input directory', type=str)
parser.add_argument('--persistent_workers', action='store_true', default=False)
parser.add_argument('--num_workers', help='Number of workers', type=int, default=2)
parser.add_argument('--no_save', action='store_true', default=False)
parser.add_argument('--monitor_test', action='store_true', default=False)
parser.add_argument('--out_suffix', help='Output suffix', default='')
parser.add_argument('--in_suffix', help='Input suffix', default='')
parser.add_argument('--frozen', action='store_true', help='freeze model during train', default=False)
parser.add_argument('--show_mem_info', action='store_true', help='Whether to show memory usage', default=False)
settings, unknown = parser.parse_known_args()
setattr(settings, 'device', torch.device('cuda'))
