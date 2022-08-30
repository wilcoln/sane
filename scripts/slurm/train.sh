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



# To run fg
# expert='results/trainers/2022-08-09_17-08-39_087977_dataset=ESNLI_model=SANENoKnowledge_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_num_attn_heads=1_no_knowledge=True'
# python src/experiments/esnli_train.py --data_frac=.05 --expert=$expert --batch_size=64 --knowledge_noise_prop=1.0
# python src/experiments/esnli_train.py --data_frac=.05 --batch_size=128 --knowledge_noise_prop=1.0


# To run bg
# python src/experiments/esnli_train.py --data_frac=1.0 --batch_size=80 # 308105

# python src/experiments/esnli_train.py --data_frac=1.0 --batch_size=80 --no_knowledge # 308137

# expert='results/trainers/2022-08-13_02-49-18_859156_dataset=ESNLI_model=SANENoKnowledge_num_epochs=5_batch_size=80_lr=0.0001_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=1.0_alpha=0.4_num_attn_heads=1_no_knowledge=True' # wait for 308034 & 35
# python src/experiments/esnli_train.py --data_frac=1.0 --expert=$expert --batch_size=64

# Maximizing knowledge relevance
# python src/experiments/train.py --data_frac=1.0 --batch_size=128 --dataset=comve
# python src/experiments/train.py --data_frac=1.0 --batch_size=128 --dataset=cose --chunk_size=5000 --num_epochs=10 --max_concepts_per_sent=20 --hidden_dim=768
# python src/experiments/train.py --data_frac=1.0 --batch_size=128 --dataset=cose --chunk_size=5000 --num_epochs=20
# python src/preprocessing/comve.py
# python src/preprocessing/cose.py --chunk_size=5000
# python src/preprocessing/esnli.py --data_frac=.05

# Test new architecture
# python src/experiments/train.py --data_frac=.05 --dataset=esnli --batch_size=64
# python src/experiments/train.py --dataset=comve --num_epochs=20 --alpha=.01 --beta=.9 --hidden_dim=512 --max_concepts_per_sent=10 --weight_decay=.1
# python src/experiments/train.py --dataset=cose --num_epochs=20 --alpha=.4 --max_concepts_per_sent=20


# Monitor all of these
# python src/experiments/train.py --dataset=comve --num_epochs=15 --algo=4
# python src/experiments/train.py --dataset=comve --num_epochs=15 --algo=2
# python src/experiments/train.py --dataset=comve --num_epochs=15 --algo=3

# python src/experiments/train.py --dataset=comve --num_epochs=15 --beta=0
python src/experiments/train.py --dataset=comve --num_epochs=15 --no_train_nk --no_gnn
# python src/experiments/train.py --dataset=cose --num_epochs=20 --hidden_dim=256 --max_concepts_per_sent=20 --algo=4
# python src/experiments/train.py --dataset=cose --num_epochs=15 --no_knowledge --chunk_size=5000
# python src/experiments/train.py --dataset=esnli --num_epochs=5 --beta=0 --batch_size=64