#!/bin/bash
#SBATCH --job-name=MNIST_train_fc
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --array=0-49
#SBATCH --output=logs_training/MNIST_train_fc_%a.out

module load StdEnv/2023
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_main.txt

TASK_ID=${SLURM_ARRAY_TASK_ID}

SEED=$((TASK_ID * 5))

echo "Train MNIST FC model with seed=$SEED"

srun python code/MNIST/train_fc.py --seed $SEED
