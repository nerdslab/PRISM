#!/bin/bash
#SBATCH --job-name=ctx336bl
#SBATCH --partition=b200-mig45,b200-mig90
#SBATCH --qos=mig-max
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/vast/projects/dyer1/lab/user/zihao/PRISM/scripts/rebuttal/logs/slurm/%x_%A_%a.out
#SBATCH --error=/vast/projects/dyer1/lab/user/zihao/PRISM/scripts/rebuttal/logs/slurm/%x_%A_%a.err
# Array over (MODEL, DATASET) pairs listed in jobs.txt (one "MODEL DATASET" per line).
set -uo pipefail
LIST=/vast/projects/dyer1/lab/user/zihao/PRISM/scripts/rebuttal/ctx336_baselines/jobs.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
export MODEL=$(echo "$LINE" | awk '{print $1}')
export DATASET=$(echo "$LINE" | awk '{print $2}')
echo "task $SLURM_ARRAY_TASK_ID -> MODEL=$MODEL DATASET=$DATASET on $(hostname)"
nvidia-smi -L || true
bash /vast/projects/dyer1/lab/user/zihao/PRISM/scripts/rebuttal/ctx336_baselines/run_one.sh
