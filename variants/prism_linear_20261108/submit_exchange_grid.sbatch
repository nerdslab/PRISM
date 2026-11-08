#!/bin/bash
#SBATCH --job-name=prismlin_exg
#SBATCH --partition=b200-mig45,b200-mig90
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/vast/projects/dyer1/lab/user/zihao/PRISM/variants/prism_linear_20261108/slurm/%x_%j.out
set -uo pipefail
mkdir -p /vast/projects/dyer1/lab/user/zihao/PRISM/variants/prism_linear_20261108/slurm
echo "$(hostname)"; nvidia-smi -L || true
bash /vast/projects/dyer1/lab/user/zihao/PRISM/variants/prism_linear_20261108/run_exchange_grid.sh
