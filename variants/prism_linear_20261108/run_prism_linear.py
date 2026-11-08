#!/usr/bin/env python
"""PRISM-Linear (2026-11-08) training entry (adapted from prism_xattn_v1_20261108/run_prism_xattn.py).

Thin wrapper around PRISM/run_prism.py: same CLI, plus
  --linear_heads {0,1}   1 = per-band heads are a single Linear(L_i, H) (DLinear form); 0 = PRISM's 2-layer heads
  --no_se {0,1}          1 = SE block over bands replaced by identity
  --run_tag              output sub-folder name (default derived from the flags)
Outputs go to variants/prism_linear_20261108/logs/<run_tag>/<dataset>_cont<L>_<features>/
and never into temp/logs/exp_prism.
"""
import argparse
import functools
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)  # run_prism uses ./temp/datasets relative paths
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DIR", os.path.join(HERE, "logs", "wandb"))

p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
p.add_argument("--linear_heads", type=int, default=1)
p.add_argument("--no_se", type=int, default=1)
p.add_argument("--run_tag", type=str, default=None)
p.add_argument("--es_delta", type=float, default=None,
               help="early-stopping threshold: an epoch counts as improvement only if val loss drops by more than this "
                    "(exp code hard-codes 2e-6); patience comes from --patience")
xa, rest = p.parse_known_args()
sys.argv = [sys.argv[0]] + rest

import torch.multiprocessing as mp  # noqa: E402
import prism_linear  # noqa: E402
import run_prism as R  # noqa: E402
import src.exp.exp_long_term_forecasting as E  # noqa: E402

TAG = xa.run_tag or (f"lin{xa.linear_heads}_nose{xa.no_se}")

E.prism_model = functools.partial(prism_linear.prism, linear_heads=bool(xa.linear_heads), no_se=bool(xa.no_se))

_orig_run_name = R.run_name_from_args


def run_dir_from_args(args) -> Path:
    return Path(HERE) / "logs" / TAG / f"{args.data}_cont{args.seq_len}_{args.features}"


def run_name_from_args(args) -> str:
    return _orig_run_name(args) + f"_{TAG}"


R.run_dir_from_args = run_dir_from_args
R.run_name_from_args = run_name_from_args

if xa.es_delta is not None:
    _BaseES = E.EarlyStopping

    class FastEarlyStopping(_BaseES):
        """Same logic as src.utils.tools.EarlyStopping, but the improvement threshold is --es_delta."""

        def __init__(self, patience=7, verbose=False, delta=0):
            super().__init__(patience=patience, verbose=verbose, delta=xa.es_delta)
            print(f"[PRISM-Linear] early stopping: patience={patience} delta={xa.es_delta}")

    E.EarlyStopping = FastEarlyStopping

if __name__ == "__main__":
    args = R.parse_args()
    print(f"[PRISM-Linear 2026-11-08] tag={TAG} linear_heads={xa.linear_heads} no_se={xa.no_se}")
    print(f"[PRISM-Linear] run_dir={run_dir_from_args(args)}")
    if args.use_multi_gpu and len(args.devices.split(",")) > 1:
        world_size = len(args.devices.split(","))
        mp.spawn(R.main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        R.main(0, 1, args)
