#!/bin/bash
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00

module load python
source .venv/bin/activate

SEED=${1:-10}  # Default to 10 if not provided
srun python code/train_baseline_model.py --seed $SEED
