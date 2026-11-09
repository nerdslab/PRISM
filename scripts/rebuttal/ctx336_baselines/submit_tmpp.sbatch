#!/bin/bash
#SBATCH --job-name=tmpp336
#SBATCH --partition=b200-mig45,b200-mig90
#SBATCH --qos=mig-max
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/vast/projects/dyer1/lab/user/zihao/PRISM/scripts/rebuttal/logs/slurm/%x_%A_%a.out
#SBATCH --error=/vast/projects/dyer1/lab/user/zihao/PRISM/scripts/rebuttal/logs/slurm/%x_%A_%a.err
# Array over datasets for the TimeMixer++ re-implementation at ctx 336.
# SMOKE=1 runs ETTh1 only, 1 epoch, horizons 96 and 720.
set -uo pipefail
DSETS=(ETTh1 ETTh2 ETTm1 ETTm2 weather exchange electricity traffic)
if [ "${SMOKE:-0}" = "1" ]; then
  export DATASET=ETTh1 HORIZONS="96 720" EPOCHS_OVERRIDE=1
else
  export DATASET=${DSETS[$SLURM_ARRAY_TASK_ID]}
fi
echo "task ${SLURM_ARRAY_TASK_ID:-smoke} -> DATASET=$DATASET on $(hostname)"; nvidia-smi -L || true
bash /vast/projects/dyer1/lab/user/zihao/PRISM/scripts/rebuttal/ctx336_baselines/run_tmpp.sh
