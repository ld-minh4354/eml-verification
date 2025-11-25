#!/bin/bash
#SBATCH --job-name=raw_result
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=1:00:00
#SBATCH --output=logs/raw_result.out

module load StdEnv/2023
module load python/3.11
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_main.txt

echo "Determining raw results"

srun python code/result_processing/raw_result.py
