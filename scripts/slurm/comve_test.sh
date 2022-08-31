#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=1:00:00

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH --partition=devel

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=wilfried.bounsi@cs.ox.ac.uk

# SANE
input_dir='results/trainers/2022-08-30_16-08-04_306433_model=SANE_dataset=comve_num_epochs=15_batch_size=128_lr=5e-05_weight_decay=0.01_sent_dim=768_'
python src/experiments/suite.py --dataset=comve --input_dir=$input_dir

sleep 5s

# NO-GNN
input_dir='results/trainers/2022-08-30_15-15-17_061483_model=SANE_dataset=comve_num_epochs=15_batch_size=128_lr=5e-05_weight_decay=0.01_sent_dim=768_'
python src/experiments/suite.py --dataset=comve --input_dir=$input_dir

# NO-KNOWLEDGE
# input_dir='results/trainers/2022-08-30_14-22-00_605716_model=SANE_dataset=esnli_num_epochs=5_batch_size=64_lr=5e-05_weight_decay=0.01_sent_dim=768_hi'
# python src/experiments/suite.py --input_dir=$input_dir