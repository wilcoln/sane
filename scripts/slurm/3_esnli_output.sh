#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=wilfried.bounsi@cs.ox.ac.uk

# run the application
input_dir=results/trainers/2022-08-06_12-44-47_683701_dataset=ESNLI_model=KAX_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=32_nle_pred=True_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4
python src/experiments/esnli_output.py --data_frac=.05 --nle_pred --input_dir=$input_dir