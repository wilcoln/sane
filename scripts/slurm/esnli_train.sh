#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=72:00:00

# set number of GPUs
#SBATCH --gres=gpu:1

# Set partition
#SBATCH --partition=devel

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=wilfried.bounsi@cs.ox.ac.uk

# Get Auto NLE CACHE
python src/utils/get_auto_nle_cache.py

# # SANE - Running
python src/experiments/train.py --dataset=esnli --num_epochs=5 --batch_size=80  --no_train_nk

# # NO-GNN - Done
# python src/experiments/train.py --dataset=esnli --num_epochs=5 --batch_size=80 --no_gnn --no_train_nk

# # NO-KNOWLEDGE - Running
# python src/experiments/train.py --dataset=esnli --num_epochs=5 --batch_size=80 --no_knowledge