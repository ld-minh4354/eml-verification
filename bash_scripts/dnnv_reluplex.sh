#!/bin/bash
#SBATCH --job-name=dnnv_reluplex
#SBATCH --mem=3G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/dnnv_reluplex.out
#SBATCH --error=logs/dnnv_reluplex.err

module load StdEnv/2020
module load python/3.9
module load scipy-stack/2021a
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_dnnv.txt

dnnv_manage install reluplex

X=$(( SLURM_ARRAY_TASK_ID ))

dnnv --reluplex \
    --prop.epsilon=0.01 \
    --network N models/MNIST/baseline/resnet18-MNIST-10.onnx \
    properties/MNIST/property_${X}.py