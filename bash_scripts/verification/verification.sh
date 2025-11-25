#!/bin/bash

CHUNK_SIZE=800
MAX_TASK_ID=7999
THROTTLE=500

start=0

while [ $start -le $MAX_TASK_ID ]; do
    end=$((start + CHUNK_SIZE - 1))
    if [ $end -gt $MAX_TASK_ID ]; then
        end=$MAX_TASK_ID
    fi

    echo "Submitting chunk: $start-$end"

    sbatch --array=${start}-${end}%${THROTTLE} bash_scripts/verification/dnnv_reluplex.sh 0.01

    start=$((end + 1))
done
