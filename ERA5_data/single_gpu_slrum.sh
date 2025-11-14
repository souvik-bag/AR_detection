#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --account=lzxvc-stat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --job-name=ar_single
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-user=sbk29@umsystem.edu
#SBATCH --mail-type=FAIL,END

# Create logs directory
# mkdir -p /mnt/pixstor/data/sbk29/logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=========================================="
echo ""

# Load environment
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate turbulence

# Verify GPU
echo "GPU Information:"
nvidia-smi
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1

# Run training
echo "Starting single GPU training..."
python single_gpu_model_train.py

echo ""
echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="