#!/bin/bash
#SBATCH --job-name=train_baseline_CIFAR10
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --array=1-10
#SBATCH --output=logs/train_baseline_CIFAR10_%a.out
#SBATCH --error=logs/train_baseline_CIFAR10_%a.err

module load StdEnv/2020
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_local.txt

SEED=$(( SLURM_ARRAY_TASK_ID * 10 ))

srun python code/train_baseline_CIFAR10.py --seed $SEED
