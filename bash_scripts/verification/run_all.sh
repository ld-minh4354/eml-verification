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
    jobid=$(sbatch --parsable --array=${start}-${end}%${THROTTLE} bash_scripts/verification/dnnv_reluplex.sh 0.01)

    echo "Waiting for chunk job $jobid to finish..."
    # Wait until no tasks of this job ID are running or pending
    while squeue -j $jobid >/dev/null 2>&1 && squeue -j $jobid | grep -q $jobid; do
        sleep 30
    done

    start=$((end + 1))
done

echo "All chunks completed."
