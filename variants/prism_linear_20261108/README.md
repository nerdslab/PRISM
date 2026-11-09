# PRISM-Linear (2026-11-08): PRISM with DLinear-style heads, Exchange study

Question (user, 2026-11-08): can PRISM beat DLinear on Exchange (DLinear paper, ctx 336: 0.081/0.203, 0.157/0.293,
0.305/0.414, 0.643/0.601) when it is put in a form similar to DLinear?

| file | role |
|---|---|
| `prism_linear.py` | subclass of the frozen PRISM copy (`../prism_xattn_v1_20261108/prism_base_20261108.py`): `linear_heads` turns every per-band head into one `Linear(L_i, H)`, `no_se` removes the SE gate over bands. Both off = exact PRISM. CPU smoke passed. |
| `run_prism_linear.py` | training entry (wraps run_prism.py; outputs under `logs/<tag>/`) |
| `run_exchange_grid.sh` | 9 configs x 4 horizons, seed 11, reference recipe of scripts/exchange_quick.sh otherwise |
| `select_exchange.py` | summary + selection by lowest **validation** MSE per horizon (rule fixed before looking at test) |
| `submit_exchange_grid.sbatch` | one GPU job, sequential |

Configs: `ref` (PRISM as in the paper), `lin_se` (linear heads, SE kept), `lin` (linear heads, no SE),
`lin_lr5e-4`, `lin_lr1e-3`, `lin_mse`, `lin_mse_lr5e-4`, `lin_L1` (depth 1), `lin_last` (level-0 heads only).

Findings while setting up: `--dropout` is stored by the model but never applied (no nn.Dropout in the tree), so the
paper's "dropout 0.5" on Exchange is a no-op. `run_prism.py`'s final line `test mae:X, mse:Y` has X = MSE (see
`../prism_xattn_v1_20261108/README.md`); `runs.txt` END lines here are already relabelled.

## Log
- 2026-11-08 16:05: grid submitted (job id in `.last_grid_jobid`).

## Result (workstation RTX 5090, 2026-11-08; logs merged here 2026-11-09)

All 9 configs x 4 horizons finished (`logs/runs.txt`, `logs/selection.txt`). Selection by lowest validation MSE:

| H | selected | test MSE / MAE | ref (unchanged PRISM) | DLinear ctx-336 |
|---|---|---|---|---|
| 96  | lin_mse_lr5e-4 | 0.086 / 0.204 | 0.083 / 0.204 | 0.081 / 0.203 |
| 192 | lin_lr1e-3     | 0.178 / 0.299 | 0.178 / 0.304 | 0.157 / 0.293 |
| 336 | lin            | 0.333 / 0.416 | 0.325 / 0.417 | 0.305 / 0.414 |
| 720 | ref            | 0.959 / 0.756 | 0.959 / 0.756 | 0.643 / 0.601 |

No configuration beats DLinear at any horizon, not even when cherry-picking the best test number (H96 best 0.082,
H192 0.174, H336 0.325, H720 0.947). At H=720 every run has its best validation loss at epoch 0 and stops after the
25-epoch patience; training only moves the model away from the near-persistence solution that DLinear converges to.
Linear heads / no-SE change exchange by at most 0.01 at H<=336 and do not address the H=720 failure. Also note that
the unchanged `ref` config gives 0.959 at H=720 versus 0.712 in Table 1 (seed 11 here, Table 1 provenance unclear).
