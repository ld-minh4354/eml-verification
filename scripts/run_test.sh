#!/bin/bash
#SBATCH –J run_test
#SBATCH –N 1 --ntasks-per-node=1
#SBATCH –-cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH –t 00:05:00

srun python test.py
