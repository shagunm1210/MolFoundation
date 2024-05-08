#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=16384
#SBATCH --job-name=MD_trial_mpi2
#SBATCH --output=log_trial.txt

# example module load
module load cuda

# example conda env activation
source /opt/python/3.8a/bin/activate
conda activate myenv

# run code here