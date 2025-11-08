#!/bin/bash
#SBATCH --job-name=prune_MNIST
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --array=0-0
#SBATCH --output=logs/prune_MNIST_%a.out
#SBATCH --error=logs/prune_MNIST_%a.err

module load python
source .venv/bin/activate

SEED_VALUES=(10 20 30 40 50 60 70 80 90 100)
PRUNE_VALUES=(10 20 30 40 50)

# Compute i, j indices from SLURM_ARRAY_TASK_ID
NUM_J=${#PRUNE_VALUES[@]}
seed_index=$(( SLURM_ARRAY_TASK_ID / NUM_J ))
prune_index=$(( SLURM_ARRAY_TASK_ID % NUM_J ))

seed=${SEED_VALUES[$seed_index]}
prune=${PRUNE_VALUES[$prune_index]}

echo "Prune MNIST with seed=$seed, prune=$prune"

srun python code/prune_MNIST.py --seed $seed --prune $prune