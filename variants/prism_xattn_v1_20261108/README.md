# PRISM-X v1 — cross-channel context for PRISM (2026-11-08)

Self-contained variant folder. Nothing in `src/`, `scripts/` or `temp/logs/exp_prism` is touched.

| file | role |
|---|---|
| `prism_base_20261108.py` | frozen copy of `src/arch/prism.py` as of 2026-11-08 (the model behind Table 1) |
| `prism_xattn.py` | the variant: subclasses of the frozen copy + the new modules; `python prism_xattn.py` runs a CPU smoke test |
| `run_prism_xattn.py` | training entry, wraps `run_prism.py`; `--xattn 0/1`, `--xattn_mode`, `--run_tag` |
| `run_xattn_one.sh` | one run with the Table 1 hyperparameters of `scripts/traffic_quick.sh` / `electricity_quick.sh` |
| `submit_xattn.sbatch` + `jobs.txt` | slurm array (MODE DATASET H per line) |
| `logs/<tag>/<dataset>_cont336_M/*.log` | training logs; `logs/runs.txt` start/end lines with the test MSE/MAE |

## What the variant adds (modelled on SCRYER, `scryer/src/arch/scryer.py` + `layers/selector.py`)

1. **Channel context.** The root-level Haar decomposition of the normalised context gives the same 6 band
   statistics PRISM's routers use. Mean over the K bands -> LayerNorm -> Linear(6, 128) -> SquaredGELU, plus a
   learned channel-identity embedding (SCRYER's unit embedding). Then one pre-LN multi-head self-attention block
   with FFN over the C channels and a second attention block with LayerNorm residual (SCRYER's
   `enable_channel_cross_attn`). Output: one 128-d vector per channel per sample.
2. **Alpha path (selection).** The context, projected to 16 dims, is concatenated to the 6 statistics at every
   router of the tree, so band and sibling softmaxes can depend on the other channels. The new input columns are
   zero-initialised, so training starts from PRISM's routing.
3. **FiLM path.** `y = y * (1 + alpha) + beta` on the per-channel forecast in normalised space, `alpha, beta =
   MLP(context)` with zero-initialised last layers (SCRYER `film_alpha_gen` / `film_beta_gen`).

With `--xattn 0` the network is exactly the frozen PRISM copy (CPU check: max |on - off| at init = 0.0).
Extra parameters: 0.33 M (C=7) to 0.44 M (C=862, traffic), on top of PRISM's 0.85–1.3 M at hidden 256.

## Protocol

Datasets: traffic (C=862) and electricity (C=321) only, context 336, horizons 96/192/336/720, seed 15,
hyperparameters as in the quick scripts (hidden 256, dropout 0.4, K=5, depth 2, overlap 8, MAE loss, lr 1e-4,
lradj 1, AMP, patience 15, up to 500 epochs; batch 64 / 128). One GPU instead of the original two.
Both `base` (frozen PRISM, same seed) and `xattn_full` are run so the comparison is same-code, same-seed.

## Log of changes

- 2026-11-08 v1: created; CPU smoke passed; GPU smoke (1 epoch) and full 16-run array submitted.
- 2026-11-08 07:30: GPU smoke 10216346 passed (3/3, rc=0; 1-epoch traffic H96 xattn MSE 0.346 vs base 0.387, electricity xattn 0.287).
  Added `--es_delta` (fast early stopping: an epoch counts as improvement only if val loss drops by more than the
  threshold). Traffic array 10217484 (xattn + base x 4 horizons) submitted with PATIENCE=3 ES_DELTA=5e-4;
  tags `*_fast_p3_d5e-4`. Electricity not yet submitted.
- 2026-11-08 10:30: NOTE on `logs/runs.txt` END lines: `run_prism.py` does `mae, mse = exp.test()` while `exp.test()`
  returns `(mse, mae)`, so the line `test mae:X, mse:Y` has X = MSE and Y = MAE. The per-epoch
  `[epoch k] test_mse=..., test_mae=...` lines in the logs are correctly labelled; use those.
  Sanity check: the 1-GPU `base_fast` H96 run matches the original 2-GPU Table-1 log
  (temp/logs/exp_prism/traffic_cont336_M/pred96_..._p15.log) epoch by epoch (epoch 28: MSE 0.4070 vs 0.4068).

## Results: traffic, fast stop (patience 3, delta 5e-4), seed 15, 1 GPU — 2026-11-08

Test MSE / MAE from the per-epoch log lines (stop epoch in brackets).

| H | base | xattn_full |
|---|---|---|
| 96 | 0.407 / 0.263 (28) | 0.409 / 0.267 (15) |
| 192 | 0.415 / 0.267 (33) | 0.417 / 0.273 (20) |
| 336 | 0.427 / 0.275 (31) | 0.442 / 0.288 (26) |
| 720 | running at commit time, 0.458 / 0.299 at epoch 16 | 0.462 / 0.300 (19) |

xattn is ahead for the first ~5 epochs, then reaches the same plateau and stops earlier. No gain on traffic.
The 1-GPU base matches the original 2-GPU Table-1 log epoch by epoch. See NOTES_vs_WDformer_20261108.md for why.
