#!/bin/bash
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mail-user=minhle@gatech.edu
#SBATCH --mail-type=ALL

module load python

SEED=${1:-10}  # Default to 10 if not provided
srun python baseline_model/training.py --seed $SEED
