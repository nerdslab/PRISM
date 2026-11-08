#!/usr/bin/env python
"""PRISM-X v1 (2026-11-08) training entry.

Thin wrapper around PRISM/run_prism.py: same CLI, plus
  --xattn {0,1}          0 = frozen PRISM copy (baseline), 1 = cross-channel context on
  --xattn_mode           full | router_only | film_only
  --xattn_d / --xattn_emb / --xattn_heads / --xattn_dropout / --xattn_identity
  --run_tag              output sub-folder name (default derived from the flags)
Outputs go to variants/prism_xattn_v1_20261108/logs/<run_tag>/<dataset>_cont<L>_<features>/
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

p = argparse.ArgumentParser(add_help=False)
p.add_argument("--xattn", type=int, default=1)
p.add_argument("--xattn_mode", type=str, default="full", choices=["full", "router_only", "film_only"])
p.add_argument("--xattn_d", type=int, default=128)
p.add_argument("--xattn_emb", type=int, default=16)
p.add_argument("--xattn_heads", type=int, default=4)
p.add_argument("--xattn_dropout", type=float, default=0.1)
p.add_argument("--xattn_identity", type=int, default=1)
p.add_argument("--run_tag", type=str, default=None)
p.add_argument("--es_delta", type=float, default=None,
               help="early-stopping threshold: an epoch counts as improvement only if val loss drops by more than this "
                    "(exp code hard-codes 2e-6); patience comes from --patience")
xa, rest = p.parse_known_args()
sys.argv = [sys.argv[0]] + rest

import torch.multiprocessing as mp  # noqa: E402
import prism_xattn  # noqa: E402
import run_prism as R  # noqa: E402
import src.exp.exp_long_term_forecasting as E  # noqa: E402

TAG = xa.run_tag or (f"xattn_{xa.xattn_mode}_d{xa.xattn_d}_e{xa.xattn_emb}" if xa.xattn else "base")

E.prism_model = functools.partial(
    prism_xattn.prism, use_channel_attn=bool(xa.xattn), xattn_mode=xa.xattn_mode, xattn_d=xa.xattn_d,
    xattn_emb=xa.xattn_emb, xattn_heads=xa.xattn_heads, xattn_dropout=xa.xattn_dropout,
    xattn_identity=bool(xa.xattn_identity))

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
            print(f"[PRISM-X] early stopping: patience={patience} delta={xa.es_delta}")

    E.EarlyStopping = FastEarlyStopping

if __name__ == "__main__":
    args = R.parse_args()
    print(f"[PRISM-X v1 2026-11-08] tag={TAG} xattn={xa.xattn} mode={xa.xattn_mode} d={xa.xattn_d} "
          f"emb={xa.xattn_emb} heads={xa.xattn_heads} dropout={xa.xattn_dropout} identity={xa.xattn_identity}")
    print(f"[PRISM-X] run_dir={run_dir_from_args(args)}")
    if args.use_multi_gpu and len(args.devices.split(",")) > 1:
        world_size = len(args.devices.split(","))
        mp.spawn(R.main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        R.main(0, 1, args)
