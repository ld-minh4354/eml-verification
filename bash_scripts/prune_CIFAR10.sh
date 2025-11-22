#!/bin/bash
#SBATCH --job-name=prune_CIFAR10
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=10:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/prune_CIFAR10_%a.out
#SBATCH --error=logs/prune_CIFAR10_%a.err

module load StdEnv/2023
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_main.txt

SEED_VALUES=(10 20 30 40 50 60 70 80 90 100)
PRUNE_VALUES=(20 40 60)

NUM_J=${#PRUNE_VALUES[@]}
seed_index=$(( SLURM_ARRAY_TASK_ID / NUM_J ))
prune_index=$(( SLURM_ARRAY_TASK_ID % NUM_J ))

seed=${SEED_VALUES[$seed_index]}
prune=${PRUNE_VALUES[$prune_index]}

echo "Prune CIFAR10 with seed=$seed, prune=$prune"

srun python code/prune_CIFAR10.py --seed $seed --prune $prune