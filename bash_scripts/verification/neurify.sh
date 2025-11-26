#!/bin/bash
#SBATCH --job-name=dnnv_neurify
#SBATCH --mem=3G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/neurify.out

module load StdEnv/2020
module load python/3.9
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r $HOME/requirements_dnnv.txt

dnnv_manage install neurify

X=$(( SLURM_ARRAY_TASK_ID ))

dnnv --neurify \
    --prop.epsilon=0.01 \
    --network N models/MNIST/baseline/resnet18-MNIST-10.onnx \
    properties/MNIST/property_${X}.py