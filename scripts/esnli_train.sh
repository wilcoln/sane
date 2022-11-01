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

# # SANE
# with regret - running
# python src/experiments/train.py --dataset=esnli --bart_version=large --num_epochs=5 --batch_size=16

# without regret - running
python src/experiments/train.py --dataset=esnli --bart_version=large --num_epochs=10 --batch_size=32  --no_train_nk

# # NO-GNN
# python src/experiments/train.py --dataset=esnli --bart_version=large --num_epochs=5 --batch_size=80 --no_gnn --no_train_nk

# # NO-KNOWLEDGE
# python src/experiments/train.py --dataset=esnli --bart_version=large --num_epochs=5 --batch_size=80 --no_knowledge