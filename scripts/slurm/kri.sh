#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=1:00:00

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH --partition=small

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=wilfried.bounsi@cs.ox.ac.uk

input_dir='results/trainers/2022-08-24_14-55-33_495327_model=SANE_dataset=comve_num_epochs=15_batch_size=128_algo=2_lr=0.0001_weight_decay=0.01_sent_'
python src/experiments/knowledge_relevance_index.py --input_dir=$input_dir --dataset=comve

