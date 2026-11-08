#!/bin/bash
# PRISM-X v1 (2026-11-08): one run.  Usage: MODE=xattn|base DATASET=traffic|electricity H=96 bash run_xattn_one.sh
# Optional: EPOCHS (override 500), XMODE (full|router_only|film_only), SEED (15), EXTRA (extra args)
# Hyperparameters copied from scripts/traffic_quick.sh / electricity_quick.sh (the runs behind Table 1),
# except that we use ONE GPU (the originals used 2 with DDP).
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$ROOT/../tsdiff310/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
MODE="${MODE:?}"; DATASET="${DATASET:?}"; H="${H:?}"
SEED="${SEED:-15}"; EPOCHS="${EPOCHS:-500}"; XMODE="${XMODE:-full}"
# Fast early stopping (user, 2026-11-08): stop when val loss fails to drop by more than ES_DELTA for PATIENCE epochs.
PATIENCE="${PATIENCE:-15}"; ES_DELTA="${ES_DELTA:-}"
export MASTER_ADDR=localhost MASTER_PORT=$((20000 + RANDOM % 10000))

case "$DATASET" in
  traffic)     C=862; bs=64;  root=./temp/datasets/traffic;     csv=traffic.csv ;;
  electricity) C=321; bs=128; root=./temp/datasets/electricity; csv=electricity.csv ;;
  *) echo "unsupported DATASET=$DATASET"; exit 1 ;;
esac
if [ "$MODE" = "xattn" ]; then XARGS="--xattn 1 --xattn_mode $XMODE"; TAG="xattn_${XMODE}_d128_e16"; else XARGS="--xattn 0"; TAG=base; fi
[ "$EPOCHS" != "500" ] && TAG="${TAG}_smoke_ep${EPOCHS}"
ESARGS=""
if [ -n "$ES_DELTA" ]; then ESARGS="--es_delta $ES_DELTA"; TAG="${TAG}_fast_p${PATIENCE}_d${ES_DELTA}"; fi
OUT="$HERE/logs/$TAG/${DATASET}_cont336_M"; mkdir -p "$OUT"
LOG="$OUT/pred${H}_cont336_s${SEED}_M_w5_L2_ov8_d0.4_lossmae_p${PATIENCE}.log"
echo "[$(date '+%F %T')] START $TAG $DATASET H=$H seed=$SEED -> $LOG" | tee -a "$HERE/logs/runs.txt"
cd "$ROOT"
python -u "$HERE/run_prism_xattn.py" $XARGS $ESARGS --run_tag "$TAG" \
  --data $DATASET --root_path $root --data_path $csv --target OT --features M --num_channels $C \
  --seq_len 336 --pred_len $H --enc_hidden 256 --dec_hidden 256 --dropout 0.4 --num_components 5 \
  --train_epochs $EPOCHS --batch_size $bs --patience $PATIENCE --lr 1e-4 --loss mae --lradj 1 --use_amp \
  --devices 0 --seed $SEED --tree_depth 2 --overlap 8 ${EXTRA:-} > "$LOG" 2>&1
rc=$?
res=$(grep -E "^test mae:" "$LOG" | tail -1)
echo "[$(date '+%F %T')] END   $TAG $DATASET H=$H seed=$SEED rc=$rc $res" | tee -a "$HERE/logs/runs.txt"
