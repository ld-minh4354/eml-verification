#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --mem=3G
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --output=logs/test_model.out
#SBATCH --error=logs/test_model.err

module load StdEnv/2020
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_dnnv.txt

srun python code/test_model.py