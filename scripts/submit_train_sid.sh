#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=69GB
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=iaifi_lab
#SBATCH -p iaifi_gpu

source ~/.bashrc

module load Anaconda3/2022.05
module load gcc/8.2.0-fasrc01
module load cuda/11.7.1-fasrc01
module load glib/2.56.1-fasrc01
module load openmpi/4.0.1-fasrc01
module load git/2.17.0-fasrc01
module load node/6.10.1-fasrc01
module load OpenBLAS/0.3.7-fasrc01

cd /n/holystore01/LABS/iaifi_lab/Users/smsharma/set-diffuser/

python -u train.py --config ./configs/nbody_sid.py
# python -u train.py --config ./configs/jets.py
