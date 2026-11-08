#!/bin/bash
# PRISM-X v2 (2026-11-08): one run with the WDformer training recipe by default.
# Usage: MODE=x2|base DATASET=traffic|electricity H=96 bash run_x2_one.sh
# Recipe env (defaults = WDformer traffic script): LOSS=mse LR=1e-3 BS=16 EPOCHS=10 PATIENCE=3 LR_HALVE=1
# Model env: TOKEN=both|wave|stats  D=256  LAYERS=2  READOUT=1  XMODE=full  SEED=15  EXTRA
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$ROOT/../tsdiff310/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
MODE="${MODE:?}"; DATASET="${DATASET:?}"; H="${H:?}"
SEED="${SEED:-15}"; EPOCHS="${EPOCHS:-10}"; XMODE="${XMODE:-full}"
LOSS="${LOSS:-mse}"; LR="${LR:-1e-3}"; BS="${BS:-16}"; PATIENCE="${PATIENCE:-3}"; LR_HALVE="${LR_HALVE:-1}"
TOKEN="${TOKEN:-both}"; D="${D:-256}"; LAYERS="${LAYERS:-2}"; READOUT="${READOUT:-1}"
export MASTER_ADDR=localhost MASTER_PORT=$((20000 + RANDOM % 10000))
case "$DATASET" in
  traffic)     C=862; root=./temp/datasets/traffic;     csv=traffic.csv ;;
  electricity) C=321; root=./temp/datasets/electricity; csv=electricity.csv ;;
  *) echo "unsupported DATASET=$DATASET"; exit 1 ;;
esac
RECIPE="${LOSS}_lr${LR}_bs${BS}_ep${EPOCHS}_p${PATIENCE}_h${LR_HALVE}"
if [ "$MODE" = "x2" ]; then
  XARGS="--xattn 1 --xattn_mode $XMODE --xattn_token $TOKEN --xattn_d $D --xattn_layers $LAYERS --xattn_readout $READOUT"
  TAG="x2_${TOKEN}_d${D}_L${LAYERS}_r${READOUT}_${RECIPE}"
else
  XARGS="--xattn 0"; TAG="base_${RECIPE}"
fi
OUT="$HERE/logs/$TAG/${DATASET}_cont336_M"; mkdir -p "$OUT"
LOG="$OUT/pred${H}_cont336_s${SEED}.log"
echo "[$(date '+%F %T')] START $TAG $DATASET H=$H seed=$SEED -> $LOG" | tee -a "$HERE/logs/runs_v2.txt"
cd "$ROOT"
python -u "$HERE/run_prism_xattn_v2.py" $XARGS --lr_halve $LR_HALVE --run_tag "$TAG" \
  --data $DATASET --root_path $root --data_path $csv --target OT --features M --num_channels $C \
  --seq_len 336 --pred_len $H --enc_hidden 256 --dec_hidden 256 --dropout 0.4 --num_components 5 \
  --train_epochs $EPOCHS --batch_size $BS --patience $PATIENCE --lr $LR --loss $LOSS --lradj 1 --use_amp \
  --devices 0 --seed $SEED --tree_depth 2 --overlap 8 ${EXTRA:-} > "$LOG" 2>&1
rc=$?
# NOTE: run_prism.py prints "test mae:X, mse:Y" with the two values swapped (X is the MSE). Relabel here.
res=$(grep -E "^test mae:" "$LOG" | tail -1 | sed -E 's/test mae:([0-9.e-]+), mse:([0-9.e-]+)/MSE=\1 MAE=\2/')
ep=$(grep -oE "^\[epoch [0-9]+\]" "$LOG" | tail -1)
echo "[$(date '+%F %T')] END   $TAG $DATASET H=$H seed=$SEED rc=$rc $res last=$ep" | tee -a "$HERE/logs/runs_v2.txt"
