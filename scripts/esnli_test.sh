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

# SANE
input_dir='results/trainers/2022-09-23_20-51-46_214265_model=SANE_dataset=esnli_bart_version=large_num_epochs=5_batch_size=32_lr=5e-05_weight_decay=0'
python src/experiments/suite.py --bart_version=large --input_dir=$input_dir

# # NO-GNN - Done
# input_dir='results/trainers/2022-08-30_14-22-00_605716_model=SANE_dataset=esnli_num_epochs=5_batch_size=64_lr=5e-05_weight_decay=0.01_sent_dim=768_hi'
# python src/experiments/suite.py --input_dir=$input_dir --no_gnn

# # NO-KNOWLEDGE
# input_dir='results/trainers/2022-08-31_16-41-55_874312_model=SANENoKnowledge_dataset=esnli_num_epochs=5_batch_size=80_lr=5e-05_weight_decay=0.01_sent'
# python src/experiments/suite.py --input_dir=$input_dir --no_knowledge
