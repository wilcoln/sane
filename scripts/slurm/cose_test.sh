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

# Get Auto NLE CACHE
python src/utils/get_auto_nle_cache.py

# SANE - Done
input_dir='results/trainers/2022-08-30_16-08-51_582298_model=SANE_dataset=cose_num_epochs=15_batch_size=128_lr=5e-05_weight_decay=0.01_sent_dim=768_h'
python src/experiments/suite.py --dataset=cose --input_dir=$input_dir --chunk_size=5000

# NO-GNN - Done
input_dir='results/trainers/2022-08-30_15-15-19_607626_model=SANE_dataset=cose_num_epochs=15_batch_size=128_lr=5e-05_weight_decay=0.01_sent_dim=768_h'
python src/experiments/suite.py --dataset=cose --input_dir=$input_dir --chunk_size=5000 --no_gnn

# NO-KNOWLEDGE - Running
input_dir='results/trainers/2022-08-31_16-33-03_673181_model=SANENoKnowledge_dataset=cose_num_epochs=15_batch_size=128_lr=5e-05_weight_decay=0.01_sen'
python src/experiments/suite.py --dataset=cose --input_dir=$input_dir --chunk_size=5000 --no_knowledge
