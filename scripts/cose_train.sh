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

# # SANE - Done
# python src/experiments/train.py --dataset=cose --chunk_size=5000 --num_epochs=15  --no_train_nk

# # NO-GNN - Done
# python src/experiments/train.py --dataset=cose --chunk_size=5000 --num_epochs=15 --no_gnn --no_train_nk

# # NO-KNOWLEDGE - Running
# python src/experiments/train.py --dataset=cose --chunk_size=5000 --num_epochs=15 --no_knowledge

# With Bart-Large

# SANE - Done
python src/experiments/train.py --dataset=cose --bart_version=large --chunk_size=5000 --num_epochs=15  --no_train_nk --batch_size=64

# NO-GNN - Done
# python src/experiments/train.py --dataset=cose --bart_version=large --chunk_size=5000 --num_epochs=15 --no_gnn --no_train_nk --batch_size=64

# NO-KNOWLEDGE - Running
# python src/experiments/train.py --dataset=cose --bart_version=large --chunk_size=5000 --num_epochs=15 --no_knowledge --batch_size=64