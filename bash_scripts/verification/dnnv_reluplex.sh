#!/bin/bash
#SBATCH --job-name=dnnv_reluplex
#SBATCH --mem=3G
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=0-399
#SBATCH --output=logs/dnnv_%A_%a.out

### Environment setup

module load StdEnv/2020
module load python/3.9

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r $HOME/requirements_dnnv.txt

dnnv_manage install reluplex

### Defining variables

DATASET_VALUES=("MNIST" "CIFAR10")
MODEL_TYPE_VALUES=("baseline" "prune_0.2" "prune_0.4" "prune_0.6")
SEED_VALUES=(10 20 30 40 50 60 70 80 90 100)
PROPERTY_VALUES=($(seq 0 99))

ND=${#DATASET_VALUES[@]}        # 2
NM=${#MODEL_TYPE_VALUES[@]}     # 4
NS=${#SEED_VALUES[@]}           # 10
NP=${#PROPERTY_VALUES[@]}       # 100

TOTAL=$(( ND * NM * NS * NP ))  # 8000

TASKS_PER_ARRAY=20
START=$(( SLURM_ARRAY_TASK_ID * TASKS_PER_ARRAY ))
END=$(( START + TASKS_PER_ARRAY - 1 ))

# Avoid exceeding 8000
if (( END >= TOTAL )); then
    END=$(( TOTAL - 1 ))
fi

### Loop through 20 tasks in this array job

for (( X=$START; X<=$END; X++ )); do

    # Decode X into dataset/model/seed/property
    d=$(( X / (NM * NS * NP) ))
    m=$(( (X / (NS * NP)) % NM ))
    s=$(( (X / NP) % NS ))
    p=$(( X % NP ))

    DATASET=${DATASET_VALUES[$d]}
    MODEL=${MODEL_TYPE_VALUES[$m]}
    SEED=${SEED_VALUES[$s]}
    PROP=${PROPERTY_VALUES[$p]}

    LOGFILE="logs_verification/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${X}.out"

    echo "Running task X=${X} → log: $LOGFILE"

    {
        echo "ARRAY_ID: $SLURM_ARRAY_TASK_ID"
        echo "GLOBAL_INDEX (X): $X"
        echo "DATASET: $DATASET"
        echo "MODEL: $MODEL"
        echo "SEED: $SEED"
        echo "PROPERTY: $PROP"
        echo "VERIFIER: reluplex"
        echo "EPSILON: $1"
        echo ""

        ### Run the actual verification with 1-hour timeout
        
        timeout 1h dnnv --reluplex \
            --prop.epsilon="$1" \
            --network N models/${DATASET}/${MODEL}/resnet18-${DATASET}-${SEED}.onnx \
            properties/${DATASET}/property_${PROP}.py

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 124 ]; then
            echo ""
            echo "======================================="
            echo " TIMEOUT: Task exceeded 1 hour and was killed"
            echo "======================================="
        else
            echo ""
            echo "Task completed with exit code: $EXIT_CODE"
        fi

    } &> "$LOGFILE"

done
