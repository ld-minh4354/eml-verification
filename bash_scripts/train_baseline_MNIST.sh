#!/bin/bash
#SBATCH --job-name=train_baseline_MNIST
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --array=1-10
#SBATCH --output=logs/train_baseline_MNIST_%A_%a.out
#SBATCH --error=logs/train_baseline_MNIST_%A_%a.err

module load python
source .venv/bin/activate

SEED=$(( SLURM_ARRAY_TASK_ID * 10 ))

echo "Running with SEED=$SEED on task ID $SLURM_ARRAY_TASK_ID"
srun python code/train_baseline_MNIST.py --seed $SEED
