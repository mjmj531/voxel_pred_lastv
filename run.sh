#! /bin/bash

#SBATCH --partition=hgx
#SBATCH --job-name=voxel
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time 24:00:00

python model_training_1210.py