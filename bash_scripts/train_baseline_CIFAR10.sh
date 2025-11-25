#!/bin/bash
#SBATCH --job-name=train_baseline_CIFAR10
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --array=1-10
#SBATCH --output=logs_training/train_baseline_CIFAR10_%a.out

module load StdEnv/2023
module load python/3.11
module load scipy-stack/2025a
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_main.txt

SEED=$(( SLURM_ARRAY_TASK_ID * 10 ))

echo "Train baseline model for CIFAR10 with seed=$SEED"

srun python code/train_baseline_CIFAR10.py --seed $SEED
