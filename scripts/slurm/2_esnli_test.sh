#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH --partition=small

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=wilfried.bounsi@cs.ox.ac.uk

# run the application
# input_dir='results/trainers/2022-08-08_20-15-18_315939_dataset=ESNLI_model=KAX_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_num_attn_heads=1'
# python src/experiments/esnli_test.py --data_frac=1.0 --input_dir=$input_dir --out_suffix=2

input_dir='results/trainers/2022-08-08_20-14-58_357105_dataset=ESNLI_model=KAX_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=64_nle_pred=True_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_num_attn_heads=1'
python src/experiments/esnli_test.py --data_frac=.05 --nle_pred --input_dir=$input_dir