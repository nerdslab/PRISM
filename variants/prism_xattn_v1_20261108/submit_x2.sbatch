#!/bin/bash
#SBATCH --job-name=prismx2
#SBATCH --partition=b200-mig45,b200-mig90
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/vast/projects/dyer1/lab/user/zihao/PRISM/variants/prism_xattn_v1_20261108/slurm/%x_%A_%a.out
#SBATCH --error=/vast/projects/dyer1/lab/user/zihao/PRISM/variants/prism_xattn_v1_20261108/slurm/%x_%A_%a.err
# PRISM-X v2 (2026-11-08). Array over lines of jobs_x2.txt: "MODE DATASET H".
# SMOKE=1: 1 epoch.  Submit e.g.:  sbatch --array=0-15 submit_xattn.sbatch
set -uo pipefail
HERE=/vast/projects/dyer1/lab/user/zihao/PRISM/variants/prism_xattn_v1_20261108
LIST=${JOBS:-$HERE/jobs_x2.txt}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
export MODE=$(echo "$LINE" | awk '{print $1}') DATASET=$(echo "$LINE" | awk '{print $2}') H=$(echo "$LINE" | awk '{print $3}')
[ "${SMOKE:-0}" = "1" ] && export EPOCHS=1
echo "task $SLURM_ARRAY_TASK_ID -> $LINE on $(hostname)"; nvidia-smi -L || true
bash $HERE/run_x2_one.sh
