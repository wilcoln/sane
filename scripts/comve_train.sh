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

# With Bart-Large

# # SANE 
# with regret - running
# python src/experiments/train.py --bart_version=large --dataset=comve --num_epochs=15 --patience=15 --batch_size=64

# without regret - running
# python src/experiments/train.py --bart_version=large --dataset=comve --num_epochs=15  --no_train_nk --patience=15

# # NO-GNN - Done
# python src/experiments/train.py --bart_version=large --dataset=comve --num_epochs=15 --no_gnn --no_train_nk --patience=15

# # NO-KNOWLEDGE - Running
# python src/experiments/train.py --bart_version=large --dataset=comve --num_epochs=15 --no_knowledge --patience=15
