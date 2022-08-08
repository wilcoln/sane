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

# run the application
# python src/experiments/esnli_train.py --nle_pred
# python src/experiments/esnli_train.py

python src/experiments/esnli_train.py --data_frac=.05 --num_attn_heads=3 #! failed multihead attention doesn't work
# python src/experiments/esnli_train.py --data_frac=.05 --nle_pred --num_attn_heads=3  #! failed
# python src/experiments/esnli_train.py --data_frac=.05
# python src/experiments/esnli_train.py --data_frac=.05 --nle_pred
