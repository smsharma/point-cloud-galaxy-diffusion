#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=iaifi_lab
##SBATCH -p gpu_requeue
#SBATCH -p iaifi_gpu

export TF_CPP_MIN_LOG_LEVEL="2"

# Load modules
module load python/3.10.9-fasrc01
module load cuda/12.2.0-fasrc01
module load gcc/12.2.0-fasrc01
module load openmpi/4.1.4-fasrc01

# Activate env
mamba activate jax

# Go to dir and train
cd /n/holystore01/LABS/iaifi_lab/Users/smsharma/set-diffuser/
python -u train.py --config ./configs/nbody.py