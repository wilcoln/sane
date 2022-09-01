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

# python src/preprocessing/comve.py
# python src/preprocessing/cose.py --chunk_size=5000
# python src/preprocessing/esnli.py --data_frac=.05
python src/preprocessing/esnli.py