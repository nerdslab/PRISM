#!/bin/bash
#SBATCH --job-name=prismlin_ctl
#SBATCH --account=dyer1-international-brai
#SBATCH --qos=mig
#SBATCH --partition=b200-mig45,b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/vast/projects/dyer1/lab/user/zihao/PRISM/variants/prism_linear_20261108/slurm/ctl_%j.out
# Exchange H=720 controls (2026-11-09): PRISM-Linear x2 (const gate / learned gate, DLinear recipe) then
# DLinear with and without instance norm. All four are minutes each.
set -uo pipefail
P=/vast/projects/dyer1/lab/user/zihao/PRISM
bash $P/variants/prism_linear_20261108/run_h720_controls.sh
bash $P/variants/dlinear_instnorm_20261109/run_h720.sh
echo "ALL DONE $(date)"
