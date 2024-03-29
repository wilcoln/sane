import argparse
import os.path as osp

import torch

parser = argparse.ArgumentParser()

# Experiment settings
_exp_settings = [
    ('dataset', 'Dataset', str, 'esnli'),
    ('no_gnn', 'Disable GNN', bool, False),
    ('bart_version', 'BART version', str, 'base'),
    ('num_epochs', 'Number of epochs', int, 5),
    ('batch_size', 'Batch size', int, 128),
    ('lr', 'Learning rate', float, 5e-5),
    ('weight_decay', 'Weight Decay', float, 1e-2),
    ('no_train_nk', 'Train No Knowledge model', bool, False),
    ('hidden_dim', 'Hidden dimension', int, 64),
    ('max_concepts_per_sent', 'Max concepts per sentence', int, 200),
    ('max_length', 'Max Number of tokens per sentence', int, 512),
    ('sentence_pool', 'Sentence pool', str, 'mean'),
    ('data_frac', 'Data fraction', float, 1.0),
    ('alpha', 'NLE loss weight in total loss', float, 0.4),
    ('alpha_regret', 'NLE regret weight in total regret', float, 0.5),
    ('beta', 'Regret loss weight in regret-augmented loss', float, 0.5),
    ('num_attn_heads', 'Number of heads of the knowledge attention', int, 1),
    ('no_knowledge', 'Disable Knowledge attention', bool, False),
    ('knowledge_noise_prop', 'Knowledge Noise prop', float, 0.0),
    ('patience', 'Maximum num epochs with no progress on the val set', int, 3),
    ('num_runs', 'Number of runs', int, 10),
    ('self_loops', 'Wheteher to include self loops', bool, True),
    ('desc', 'Additional description', str, None)
]
for name, desc, type_, default in _exp_settings:
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
parser.add_argument('--show_plots', action='store_true', default=False)
parser.add_argument('--overwrite', action='store_true', default=False)
parser.add_argument('--monitor_test', action='store_true', default=False)
parser.add_argument('--frozen', action='store_true', help='freeze model during train', default=False)
parser.add_argument('--use_science', action='store_true', help='Whether to use scientific plots style', default=False)
settings, unknown = parser.parse_known_args()
setattr(settings, 'device', torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
setattr(settings, 'exp', {k: v for k, v in vars(settings).items() if k in [s[0] for s in _exp_settings] and v})
setattr(settings, 'embed_suffix', '' if settings.bart_version == 'base' else '_large')
setattr(settings, 'sent_dim', 768 if settings.bart_version == 'base' else 1024)
# Print Experimental Settings
print('*** Experimental Settings ***')
for key, value in settings.exp.items():
    print(f'{key}={value}')

# Set number of classes
num_classes = {'esnli': 3, 'comve': 2, 'cose': 5}
setattr(settings, 'num_classes', num_classes[settings.dataset])
