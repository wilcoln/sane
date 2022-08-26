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


input_dir='results/trainers/2022-08-24_04-18-51_257125_model=SANE_dataset=comve_num_epochs=20_batch_size=128_algo=2_lr=0.0001_weight_decay=0.01_sent_'
python src/experiments/loss_comparison.py --input_dir=$input_dir  --max_concepts_per_sent=20 --batch_size=1 --dataset=comve

