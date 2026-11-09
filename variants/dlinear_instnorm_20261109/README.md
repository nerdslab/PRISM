# DLinear ± instance norm on Exchange (2026-11-09)

Control for the PRISM-Linear exchange question (variants/prism_linear_20261108/README.md, "H720 controls"):
does per-window instance normalisation alone move DLinear from 0.643 to the 0.85–0.98 band where every
instance-normalised model (PatchTST, iTransformer, FilterNet, WPMixer, TimeMixer, N-HiTS, PRISM) sits at H=720?

- `run_dlinear_in.py`: executes the unmodified `external/DLinear/run_longExp.py`; with `--instnorm 1` the model
  registered as `DLinear` is wrapped in SeriesNorm-style per-window mean/std normalisation (eps 1e-5).
- `run_h720.sh`: released exchange recipe at ctx 336 (H720: bs 32, lr 5e-3 halved per epoch, 10 epochs,
  patience 3, MSE, best-val checkpoint reloaded), `INSTNORMS="0 1"`. `in0` is the sanity check against the
  literature number 0.643 / 0.601; `in1` is the control.
- Outputs: `logs/*.log`, `logs/runs.txt`, `result.txt` (written by the DLinear code), `checkpoints/`.

Nothing in external/DLinear is modified.
