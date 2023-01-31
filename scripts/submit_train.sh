#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=300GB
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=iaifi_lab
#SBATCH -p iaifi_gpu

source ~/.bashrc

cd /n/dvorkin_lab/smsharma

conda activate ddp
module load Anaconda3/2020.11
module load gcc/8.2.0-fasrc01
module load cudnn/8.2.2.26_cuda11.4-fasrc01
module load glib/2.56.1-fasrc01
module load openmpi/4.0.1-fasrc01
module load git/2.17.0-fasrc01
module load node/6.10.1-fasrc01
module load OpenBLAS/0.3.7-fasrc01

cd /n/dvorkin_lab/smsharma/functional-diffusion/

python -u train.py
