#!/bin/bash
#SBATCH --job-name=dnnv_reluplex
#SBATCH --mem=3G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --array=0-4
#SBATCH --output=logs_verification/dnnv_%A_%a.out

module load StdEnv/2020
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_dnnv.txt

dnnv_manage install reluplex

X=$(( SLURM_ARRAY_TASK_ID ))

DATASET_VALUES=("MNIST" "CIFAR10")
MODEL_TYPE_VALUES=("baseline" "prune_0.2" "prune_0.4" "prune_0.6")
SEED_VALUES=(10 20 30 40 50 60 70 80 90 100)
PROPERTY_VALUES=($(seq 0 99))

ND=${#DATASET_VALUES[@]}       # 2
NM=${#MODEL_TYPE_VALUES[@]}    # 4
NS=${#SEED_VALUES[@]}          # 10
NP=${#PROPERTY_VALUES[@]}      # 100

d=$(( $X / ( $NM * $NS * $NP ) ))
m=$(( ( $X / ( $NS * $NP )) % $NM ))
s=$(( ( $X / $NP ) % $NS ))
p=$(( $X % $NP ))

DATASET=${DATASET_VALUES[$d]}
MODEL=${MODEL_TYPE_VALUES[$m]}
SEED=${SEED_VALUES[$s]}
PROP=${PROPERTY_VALUES[$p]}

echo "DATASET: ${DATASET}"
echo "MODEL TYPE: ${MODEL}"
echo "SEED: ${SEED}"
echo "PROPERTY NO: ${PROP}"
echo "VERIFIER: reluplex"
echo "EPSILON: $1"

dnnv --reluplex \
    --prop.epsilon=$1 \
    --network N models/${DATASET}/${MODEL}/resnet18-${DATASET}-${SEED}.onnx \
    properties/${DATASET}/property_${PROP}.py