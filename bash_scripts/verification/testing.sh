#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=61G
#SBATCH --time=0:20:00
#SBATCH --output=logs/abc_testing.out

module load StdEnv/2023
module load python/3.11
source $HOME/eml-verification/.venv_abc/bin/activate

export OMP_NUM_THREADS=1

srun python $HOME/eml-verification/alpha-beta-CROWN/complete_verifier/abcrown.py \
    --config $HOME/eml-verification/alpha-beta-CROWN/complete_verifier/exp_configs/tutorial_examples/cifar_resnet_2b.yaml