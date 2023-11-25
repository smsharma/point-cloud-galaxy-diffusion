#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=iaifi_lab
#SBATCH --array=0-3
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

## Run for positions-only and all-features runs
# Set seed as the job array index
python -u infer.py --seed $SLURM_ARRAY_TASK_ID --n_steps 50 --n_elbo_samples 8 --n_test 32 --run_name "gallant-cherry-87"
python -u infer.py --seed $SLURM_ARRAY_TASK_ID --n_steps 50 --n_elbo_samples 8 --n_test 32 --run_name "magical-goosebump-109"