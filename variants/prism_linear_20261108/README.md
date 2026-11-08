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
