#!/bin/bash
#SBATCH --job-name=fill336
#SBATCH --partition=b200-mig45,b200-mig90
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/vast/projects/dyer1/lab/user/zihao/PRISM/results/rebuttal_ctx336_baselines/logs/slurm_fill_%A_%a.out
# 2026-11-08: array 0-2 -> FreqCycle_exchange, WPMixer_traffic, WPMixer_electricity
set -uo pipefail
TASKS=(FreqCycle_exchange WPMixer_traffic WPMixer_electricity)
export TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
echo "task $SLURM_ARRAY_TASK_ID -> $TASK on $(hostname)"
bash /vast/projects/dyer1/lab/user/zihao/PRISM/scripts/rebuttal/ctx336_baselines/run_fill_blanks.sh
