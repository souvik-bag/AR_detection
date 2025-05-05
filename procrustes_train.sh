#!/bin/bash

#SBATCH--partition=requeue
#SBATCH--account=general
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=procrustes_train
#SBATCH --output=procrustes_train%j.out
#SBATCH --error=procrustes_train%j.err
#SBATCH --mail-user=sbk29@umsystem.edu 
#SBATCH --mail-type=FAIL,END

# Load necessary modules
module load miniconda3

# Initialize Conda and activate the environment
eval "$(conda shell.bash hook)"
conda activate turbulence

# Run the Python script
python procrustes_training_module.py