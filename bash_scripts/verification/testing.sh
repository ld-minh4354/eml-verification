#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=61G
#SBATCH --time=0:05:00
#SBATCH --output=logs/abc_testing.out

source $HOME/eml-verification/.venv_abc/bin/activate

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

srun python $HOME/eml-verification/alpha-beta-CROWN/complete_verifier/abcrown.py \
    --config $HOME/eml-verification/alpha-beta-CROWN/complete_verifier/exp_configs/tutorial_examples/mnist_cnn_a_adv.yaml