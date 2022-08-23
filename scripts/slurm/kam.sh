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


input_dir='results/trainers/2022-08-23_07-54-20_381813_model=SANE_dataset=esnli_num_epochs=5_batch_size=64_algo=2_lr=0.0001_weight_decay=0.01_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_beta=0.5_num_attn_heads=1'
python src/experiments/knowledge_attention_map.py --input_dir=$input_dir  --max_concepts_per_sent=20 --batch_size=1

