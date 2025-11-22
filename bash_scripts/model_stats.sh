#!/bin/bash
#SBATCH --job-name=model_stats
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --array=1-10
#SBATCH --output=logs/model_stats.out
#SBATCH --error=logs/model_stats.err

module load StdEnv/2023
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_main.txt

srun python code/model_stats_CIFAR10.py
