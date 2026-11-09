#!/bin/bash
#SBATCH --job-name=bench_eff_all
#SBATCH --partition=b200-mig45,b200-mig90
#SBATCH --qos=mig-max
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=/vast/projects/dyer1/lab/user/zihao/PRISM/scripts/rebuttal/logs/slurm/%x_%j.out
#SBATCH --error=/vast/projects/dyer1/lab/user/zihao/PRISM/scripts/rebuttal/logs/slurm/%x_%j.err

# GPU efficiency benchmark: PRISM (depth 1/2) + D-PAD, DLinear, TimeMixer,
# WPMixer, xPatch, iTransformer, PatchTST, SCINet, PaiFilter, WDformer on
# ETTh1 / ctx 336 / H 96 / batch 8 / fp32.  One subprocess per model.

set -euo pipefail

REPO=/vast/projects/dyer1/lab/user/zihao/PRISM
OUT_DIR=$REPO/results/rebuttal_efficiency
mkdir -p "$OUT_DIR" "$REPO/scripts/rebuttal/logs/slurm"

source /vast/projects/dyer1/lab/user/zihao/tsdiff310/bin/activate
# ptwt (needed by WDformer) lives in the rebuttal pydeps directory
export PYTHONPATH=/vast/projects/dyer1/lab/user/zihao/PRISM/external/rebuttal_ctx336/pydeps${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

cd "$REPO"
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python scripts/rebuttal/bench_efficiency_all.py \
  --device cuda \
  --batch_size 8 \
  --warmup 50 --iters 200 \
  --train_warmup 20 --train_iters 50 \
  --out_dir "$OUT_DIR"

echo "done: $OUT_DIR/efficiency_all.csv  $OUT_DIR/efficiency_all.tex"
