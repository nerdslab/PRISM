#!/bin/bash
# PRISM-Linear exchange grid (2026-11-08). Reference recipe = scripts/exchange_quick.sh (hidden 336, MAE, bs 512,
# lr 1e-4, patience 25, 200 epochs, depth 2, overlap 8, K=5, seed 11, AMP; note: --dropout is a no-op in the model).
# One GPU, configs x horizons sequential. Selection must be done on validation (see select_exchange.py).
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
source "$ROOT/../tsdiff310/bin/activate"; export PYTHONDONTWRITEBYTECODE=1
export MASTER_ADDR=localhost MASTER_PORT=$((20000 + RANDOM % 10000))
SEED="${SEED:-11}"; EPOCHS="${EPOCHS:-200}"; HORIZONS="${HORIZONS:-96 192 336 720}"
# name | linear_heads | no_se | depth | last_only | loss | lr | hidden
CONFIGS="${CONFIGS:-
ref|0|0|2|0|mae|1e-4|336
lin_se|1|0|2|0|mae|1e-4|336
lin|1|1|2|0|mae|1e-4|336
lin_lr5e-4|1|1|2|0|mae|5e-4|336
lin_lr1e-3|1|1|2|0|mae|1e-3|336
lin_mse|1|1|2|0|mse|1e-4|336
lin_mse_lr5e-4|1|1|2|0|mse|5e-4|336
lin_L1|1|1|1|0|mae|1e-4|336
lin_last|1|1|2|1|mae|1e-4|336
}"
cd "$ROOT"
echo "$CONFIGS" | grep -v '^\s*$' | while IFS='|' read -r name lh nse depth last loss lr hidden; do
  for H in $HORIZONS; do
    TAG="exg_${name}"; OUT="$HERE/logs/$TAG/exchange_cont336_M"; mkdir -p "$OUT"
    LOG="$OUT/pred${H}_cont336_s${SEED}.log"
    [ "$last" = "1" ] && LASTARG="--use_last_layer_only" || LASTARG=""
    echo "[$(date '+%F %T')] START $TAG H=$H seed=$SEED" | tee -a "$HERE/logs/runs.txt"
    python -u "$HERE/run_prism_linear.py" --linear_heads $lh --no_se $nse --run_tag "$TAG" \
      --data exchange --root_path ./temp/datasets/exchange_rate --data_path exchange_rate.csv --target OT --features M \
      --num_channels 8 --seq_len 336 --pred_len $H --enc_hidden $hidden --dec_hidden $hidden --dropout 0.5 \
      --decomp_kind haar --num_components 5 --train_epochs $EPOCHS --batch_size 512 --patience 25 --lr $lr --loss $loss \
      --lradj 1 --use_amp --drop_last 0 --devices 0 --seed $SEED --tree_depth $depth --overlap 8 $LASTARG > "$LOG" 2>&1
    rc=$?
    res=$(grep -E "^test mae:" "$LOG" | tail -1 | sed -E 's/test mae:([0-9.e-]+), mse:([0-9.e-]+)/MSE=\1 MAE=\2/')
    echo "[$(date '+%F %T')] END   $TAG H=$H seed=$SEED rc=$rc $res epochs=$(grep -c '^\[epoch' "$LOG")" | tee -a "$HERE/logs/runs.txt"
  done
done
