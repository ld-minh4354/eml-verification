#!/bin/bash
#SBATCH --job-name=dnnv_reluplex
#SBATCH --mem=3G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --array=0-1
#SBATCH --output=logs_verification/dnnv_%A_%a.out

module load StdEnv/2020
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_dnnv.txt

dnnv_manage install reluplex

X=$(( SLURM_ARRAY_TASK_ID ))

DATASET=("MNIST" "CIFAR10")
dataset=${DATASET[$X]}

echo "DATASET: ${dataset}"
echo "MODEL TYPE: baseline"
echo "SEED: 10"
echo "PROPERTY NO: 0"
echo "VERIFIER: reluplex"

dnnv --reluplex \
    --prop.epsilon=$1 \
    --network N models/${dataset}/baseline/resnet18-${dataset}-10.onnx \
    properties/${dataset}/property_0.py