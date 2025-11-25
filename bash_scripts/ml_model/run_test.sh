#!/bin/bash
#SBATCH –N 1 --ntasks-per-node=1
#SBATCH –-cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH –-time 00:05:00

module load python
srun python test.py
