#!/bin/bash
#SBATCH --job-name=prune_MNIST
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --array=1-10
#SBATCH --output=logs/prune_MNIST_%a.out
#SBATCH --error=logs/prune_MNIST_%a.err

module load python
source .venv/bin/activate

SEED=$(( SLURM_ARRAY_TASK_ID * 10 ))

echo "Running with SEED=$SEED on task ID $SLURM_ARRAY_TASK_ID"
srun python code/prune_MNIST.py --seed $SEED
