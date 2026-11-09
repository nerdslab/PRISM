#!/bin/bash
# Exchange H=720 controls (2026-11-09), approved by the user on 11/09.
# PRISM-Linear (linear heads, no SE) trained with DLinear's exchange-720 recipe:
#   MSE loss, batch 32, lr 5e-3 halved every epoch, 10 epochs, patience 3, test the best-val weights, no AMP.
#   ctl_constgate_dlrecipe : band gates frozen to uniform  -> fixed linear map of the instance-normalised window
#   ctl_gate_dlrecipe      : PRISM's learned gates kept    -> isolates the gating from the recipe
# Usage: bash run_h720_controls.sh      (env: SEED, HORIZONS, CONFIGS)
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
source "${VENV_ACTIVATE:-$ROOT/../tsdiff310/bin/activate}"; export PYTHONDONTWRITEBYTECODE=1
export MASTER_ADDR=localhost MASTER_PORT=$((20000 + RANDOM % 10000))
SEED="${SEED:-11}"; HORIZONS="${HORIZONS:-720}"
# name | const_gate | loss | lr | bs | epochs | patience | lr_halve | load_best
CONFIGS="${CONFIGS:-
ctl_constgate_dlrecipe|1|mse|5e-3|32|10|3|1|1
ctl_gate_dlrecipe|0|mse|5e-3|32|10|3|1|1
}"
cd "$ROOT"
echo "$CONFIGS" | grep -v '^\s*$' | while IFS='|' read -r name cg loss lr bs ep pat halve lb; do
  for H in $HORIZONS; do
    TAG="$name"; OUT="$HERE/logs/$TAG/exchange_cont336_M"; mkdir -p "$OUT"
    LOG="$OUT/pred${H}_cont336_s${SEED}.log"
    echo "[$(date '+%F %T')] START $TAG H=$H seed=$SEED" | tee -a "$HERE/logs/runs_controls.txt"
    python -u "$HERE/run_prism_linear_v2.py" --linear_heads 1 --no_se 1 --const_gate $cg --lr_halve $halve --load_best $lb \
      --run_tag "$TAG" \
      --data exchange --root_path ./temp/datasets/exchange_rate --data_path exchange_rate.csv --target OT --features M \
      --num_channels 8 --seq_len 336 --pred_len $H --enc_hidden 336 --dec_hidden 336 --dropout 0.5 \
      --decomp_kind haar --num_components 5 --train_epochs $ep --batch_size $bs --patience $pat --lr $lr --loss $loss \
      --lradj 1 --drop_last 0 --devices 0 --seed $SEED --tree_depth 2 --overlap 8 > "$LOG" 2>&1
    rc=$?
    # final line of run_prism prints "test mae:X, mse:Y" with X = MSE (labels swapped in run_prism.py)
    res=$(grep -E "^test mae:" "$LOG" | tail -1 | sed -E 's/test mae:([0-9.e-]+), mse:([0-9.e-]+)/MSE=\1 MAE=\2/')
    echo "[$(date '+%F %T')] END   $TAG H=$H seed=$SEED rc=$rc $res epochs=$(grep -c '^\[epoch [0-9]*\] val_loss' "$LOG")" | tee -a "$HERE/logs/runs_controls.txt"
  done
done
