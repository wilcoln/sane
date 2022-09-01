#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=72:00:00

# set number of GPUs
#SBATCH --gres=gpu:1

# Set partition
#SBATCH --partition=small

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=wilfried.bounsi@cs.ox.ac.uk

# Get Auto NLE CACHE
python src/utils/get_auto_nle_cache.py


# # SANE - Done
# python src/experiments/train.py --dataset=comve --num_epochs=15  --no_train_nk

# # NO-GNN - Done
# python src/experiments/train.py --dataset=comve --num_epochs=15 --no_gnn --no_train_nk

# # NO-KNOWLEDGE
python src/experiments/train.py --dataset=comve --num_epochs=5 --no_knowledge