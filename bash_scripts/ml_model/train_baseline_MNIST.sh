#!/bin/bash
#SBATCH --job-name=train_baseline_MNIST
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:20:00
#SBATCH --array=1-10
#SBATCH --output=logs_training/train_baseline_MNIST_%a.out

module load StdEnv/2023
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_main.txt

SEED=$(( SLURM_ARRAY_TASK_ID * 10 ))

echo "Train baseline model for MNIST with seed=$SEED"

srun python code/ml_model/train_baseline_MNIST.py --seed $SEED
