#!/bin/bash
# DLinear exchange controls (2026-11-09): released recipe, ctx 336, with and without instance norm.
# Released exchange script: bs 8 / lr 5e-4 (H 96,192), bs 32 / lr 5e-4 (H 336), bs 32 / lr 5e-3 (H 720);
# train_epochs 10, patience 3, MSE, lradj type1, seed 0 (run_longExp default), shared (non-individual) linear.
# Usage: bash run_h720.sh    (env: HORIZONS, INSTNORMS, SEED)
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
source "${VENV_ACTIVATE:-$ROOT/../tsdiff310/bin/activate}"; export PYTHONDONTWRITEBYTECODE=1
HORIZONS="${HORIZONS:-720}"; INSTNORMS="${INSTNORMS:-0 1}"; SEED="${SEED:-0}"
DATA="$ROOT/temp/datasets/exchange_rate"
mkdir -p "$HERE/logs"; cd "$HERE"
for IN in $INSTNORMS; do for H in $HORIZONS; do
  case $H in 96|192) bs=8; lr=0.0005;; 336) bs=32; lr=0.0005;; 720) bs=32; lr=0.005;; esac
  TAG="DLinear_in${IN}_exchange_sl336_pl${H}_s${SEED}"
  echo "[$(date '+%F %T')] START $TAG" | tee -a "$HERE/logs/runs.txt"
  python -u "$HERE/run_dlinear_in.py" --instnorm $IN --is_training 1 --root_path "$DATA/" --data_path exchange_rate.csv \
    --model_id Exchange_336_${H}_in${IN} --model DLinear --data custom --features M --seq_len 336 --pred_len $H \
    --enc_in 8 --des Exp --itr 1 --batch_size $bs --learning_rate $lr --seed $SEED --num_workers 2 \
    --checkpoints "$HERE/checkpoints/" > "$HERE/logs/$TAG.log" 2>&1
  rc=$?
  res=$(grep -E "^mse:" "$HERE/logs/$TAG.log" | tail -1 | cut -d, -f1,2)
  echo "[$(date '+%F %T')] END   $TAG rc=$rc $res" | tee -a "$HERE/logs/runs.txt"
done; done
