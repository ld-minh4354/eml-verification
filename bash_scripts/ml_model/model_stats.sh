#!/bin/bash
#SBATCH --job-name=model_stats
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=1:00:00
#SBATCH --output=logs/model_stats.out

module load StdEnv/2023
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_main.txt

srun python code/ml_model/model_stats_MNIST.py
srun python code/ml_model/model_stats_CIFAR10.py
