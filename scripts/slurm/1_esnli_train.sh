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
JOB_ID=`basename "$0"`
echo 'Running Experiment '.$JOB_ID.' ...'
# python src/experiments/esnli_train.py --data_frac=1.0 --batch_size=80
python src/experiments/esnli_train.py --data_frac=.05 --batch_size=128