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

# SANE - Done
# input_dir='results/trainers/2022-08-30_16-08-04_306433_model=SANE_dataset=comve_num_epochs=15_batch_size=128_lr=5e-05_weight_decay=0.01_sent_dim=768_'
# python src/experiments/suite.py --dataset=comve --input_dir=$input_dir

# NO-GNN - Done
input_dir='results/trainers/2022-08-30_15-15-17_061483_model=SANE_dataset=comve_num_epochs=15_batch_size=128_lr=5e-05_weight_decay=0.01_sent_dim=768_'
python src/experiments/suite.py --dataset=comve --input_dir=$input_dir --no_gnn

# NO-KNOWLEDGE - Running
# input_dir='results/trainers/2022-08-31_16-46-33_830328_model=SANENoKnowledge_dataset=comve_num_epochs=5_batch_size=128_lr=5e-05_weight_decay=0.01_sen'
# python src/experiments/suite.py --dataset=comve --input_dir=$input_dir --no_knowledge
