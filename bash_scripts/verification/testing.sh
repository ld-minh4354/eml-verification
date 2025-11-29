#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=61G
#SBATCH --time=0:05:00
#SBATCH --output=logs/abc_testing.out

source $HOME/eml-verification/.venv_abc/bin/activate