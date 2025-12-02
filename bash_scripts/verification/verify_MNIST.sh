#!/bin/bash
#SBATCH --job-name=verify_MNIST
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=61G
#SBATCH --time=2:00:00
#SBATCH --array=0-0
#SBATCH --output=logs_verification/MNIST_%a.out

module load StdEnv/2023
module load python/3.11
source $HOME/eml-verification/.venv_abc/bin/activate

export OMP_NUM_THREADS=8

for (( X=0; X<20; X++ )); do
    ID=$((SLURM_ARRAY_TASK_ID * 20 + X))
    LOGFILE="logs_verification/MNIST_${ID}.out"

    {
        srun python code/property_gen/generate_property_script.py \
        --epsilon 0.01 --index $ID --job $SLURM_ARRAY_TASK_ID

        timeout 5m srun python $HOME/eml-verification/alpha-beta-CROWN/complete_verifier/abcrown.py \
        --config $HOME/eml-verification/properties/current_${SLURM_ARRAY_TASK_ID}.yaml

    } &> "$LOGFILE"
done