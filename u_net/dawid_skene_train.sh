#!/bin/bash

#SBATCH--partition=requeue
#SBATCH--account=general
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --job-name=dawid_skene_train
#SBATCH --output=dawid_skene_train%j.out
#SBATCH --error=dawid_skene_train%j.err
#SBATCH --mail-user=sbk29@umsystem.edu 
#SBATCH --mail-type=FAIL,END

# Load necessary modules
module load miniconda3

# Initialize Conda and activate the environment
eval "$(conda shell.bash hook)"
conda activate turbulence

# Run the Python script
python run_dawid_skene.py

