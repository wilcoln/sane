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

# python src/experiments/esnli_train.py --data_frac=.05 --num_attn_heads=2
# python src/experiments/esnli_train.py --data_frac=.05 --nle_pred --num_attn_heads=2

# python src/experiments/esnli_train.py --data_frac=.05
# python src/experiments/esnli_train.py --data_frac=.05 --nle_pred


# Currently running ...

# python src/experiments/esnli_train.py --data_frac=1.0 --batch_size=80
# python src/experiments/esnli_train.py --data_frac=1.0 --nle_pred --batch_size=80

# python src/experiments/esnli_train.py --data_frac=1.0 --batch_size=100 --num_attn_heads=2
# python src/experiments/esnli_train.py --data_frac=1.0 --batch_size=100 --nle_pred --num_attn_heads=2

# python src/experiments/esnli_train.py --data_frac=.05 --frozen
# python src/experiments/esnli_train.py --data_frac=.05 --no_knowledge

# python src/experiments/esnli_train.py --data_frac=.05
# python src/experiments/esnli_train.py --data_frac=.05 --knowledge_noise_prop=1.0
# python src/experiments/esnli_train.py --data_frac=.05 --knowledge_noise_prop=.8
# python src/experiments/esnli_train.py --data_frac=.05 --knowledge_noise_prop=.5
# python src/experiments/esnli_train.py --data_frac=.05 --knowledge_noise_prop=.2

expert='results/trainers/2022-08-09_17-08-39_087977_dataset=ESNLI_model=SANENoKnowledge_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_num_attn_heads=1_no_knowledge=True'
python src/experiments/esnli_train.py --data_frac=.05 --expert=$expert --batch_size=64

# python src/experiments/esnli_train.py --data_frac=.05 --batch_size=128
