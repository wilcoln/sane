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

# run the application
# input_dir='results/trainers/2022-08-08_20-15-18_315939_dataset=ESNLI_model=KAX_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_num_attn_heads=1'
# python src/experiments/esnli_test.py --data_frac=1.0 --input_dir=$input_dir --out_suffix=2

# input_dir='results/trainers/2022-08-08_20-14-58_357105_dataset=ESNLI_model=KAX_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=64_nle_pred=True_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_num_attn_heads=1'
# python src/experiments/esnli_test.py --data_frac=.05 --nle_pred --input_dir=$input_dir

# input_dir='results/trainers/2022-08-08_22-41-59_516527_dataset=ESNLI_model=KAX_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_num_attn_heads=2'
# python src/experiments/esnli_test.py --data_frac=.05 --input_dir=$input_dir --num_attn_heads=2 --out_suffix=2
# python src/experiments/esnli_test.py --data_frac=1.0 --input_dir=$input_dir --num_attn_heads=2 --out_suffix=-full2

# input_dir='results/trainers/2022-08-09_17-08-39_087977_dataset=ESNLI_model=SANENoKnowledge_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_num_attn_heads=1_no_knowledge=True'
# python src/experiments/esnli_test.py --data_frac=.05 --no_knowledge --input_dir=$input_dir
# python src/experiments/esnli_test.py --data_frac=1.0 --no_knowledge --input_dir=$input_dir --out_suffix=-full
# python tests/esnli_test.py --no_knowledge --input_dir=$input_dir --batch_size=32 --frozen


# input_dir='results/trainers/2022-08-08_22-41-59_516527_dataset=ESNLI_model=KAX_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_num_attn_heads=2'
# python tests/esnli_test.py --input_dir=$input_dir --num_attn_heads=2

# input_dir=results/trainers/2022-08-10_13-18-53_715197_dataset=ESNLI_model=SANE_num_epochs=5_batch_size=64_lr=0.0001_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_num_attn_heads=1_expert=results/trainers/2022-08-09_17-08-39_087977_dataset=ESNLI_model=SANENoKnowledge_num_epochs=5_batch_size=128_lr=0.0001_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_num_attn_heads=1_no_knowledge=True
# python tests/esnli_test.py --input_dir=$input_dir
# python src/experiments/esnli_test.py --data_frac=.05 --input_dir=$input_dir

# input_dir='results/trainers/2022-08-13_02-49-18_859156_dataset=ESNLI_model=SANENoKnowledge_num_epochs=5_batch_size=80_lr=0.0001_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=1.0_alpha=0.4_num_attn_heads=1_no_knowledge=True'
# python tests/test.py --input_dir=$input_dir --no_knowledge
# python src/experiments/test.py --data_frac=1.0 --input_dir=$input_dir --no_knowledge

# input_dir=''
# python src/experiments/test.py --data_frac=1.0 --batch_size=128 --input_dir=$input_dir --dataset=comve

# input_dir=''
# python src/experiments/test.py --data_frac=1.0 --batch_size=128 --input_dir=$input_dir --dataset=cose

# input_dir='results/trainers/2022-08-23_07-54-20_381813_model=SANE_dataset=esnli_num_epochs=5_batch_size=64_algo=2_lr=0.0001_weight_decay=0.01_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_beta=0.5_num_attn_heads=1'
# python tests/esnli_test.py --input_dir=$input_dir
# python src/experiments/test.py --input_dir=$input_dir


# input_dir='results/trainers/2022-08-23_07-54-20_381813_model=SANE_dataset=esnli_num_epochs=5_batch_size=64_algo=2_lr=0.0001_weight_decay=0.01_sent_dim=768_hidden_dim=64_max_concepts_per_sent=200_sentence_pool=mean_data_frac=0.05_alpha=0.4_beta=0.5_num_attn_heads=1'
# python src/experiments/test.py --input_dir=$input_dir

input_dir='results/trainers/2022-08-30_00-05-18_045562_model=SANE_dataset=comve_num_epochs=15_batch_size=128_algo=2_lr=0.0001_weight_decay=0.01_sent_'
python src/experiments/test.py --input_dir=$input_dir --dataset=comve --no_gnn