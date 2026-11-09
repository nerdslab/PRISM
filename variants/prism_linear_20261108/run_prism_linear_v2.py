#!/usr/bin/env python
"""PRISM-Linear v2 entry (2026-11-09): run_prism_linear.py flags plus three protocol switches used for the
exchange H=720 controls (see README, "H720 controls").

  --const_gate {0,1}   1 = every BandSelectorMLP returns uniform weights (1/K), so the internal left/right
                       competition is 0.5/0.5 and the leaf band weights are 1/K.  With --linear_heads 1 --no_se 1
                       the forecast is then a fixed linear map of the SeriesNorm-normalised window (Haar bands,
                       segment split and stitching are all linear), i.e. a DLinear-type model with instance norm.
  --lr_halve {0,1}     1 = lr <- lr0 * 0.5**epoch after every epoch (TSLib lradj=type1, what DLinear uses;
                       the exp code never calls adjust_learning_rate)
  --load_best {0,1}    1 = keep an in-memory copy of the weights at the lowest validation loss and load it
                       before test() (DLinear reloads its best checkpoint; the exp code tests the last epoch)

Everything else is identical to run_prism_linear.py.  Nothing under src/ or scripts/ is modified.
"""
import argparse
import copy
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
p.add_argument("--const_gate", type=int, default=0)
p.add_argument("--lr_halve", type=int, default=0)
p.add_argument("--load_best", type=int, default=0)
p.add_argument("--run_tag", type=str, default=None)
p.add_argument("--es_delta", type=float, default=None)
xa, rest = p.parse_known_args()
sys.argv = [sys.argv[0]] + rest

import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import prism_linear  # noqa: E402
import prism_base_20261108 as base  # noqa: E402  (the frozen PRISM copy prism_linear builds on)
import run_prism as R  # noqa: E402
import src.exp.exp_long_term_forecasting as E  # noqa: E402

TAG = xa.run_tag or (f"lin{xa.linear_heads}_nose{xa.no_se}_cg{xa.const_gate}_h{xa.lr_halve}_lb{xa.load_best}")

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
        def __init__(self, patience=7, verbose=False, delta=0):
            super().__init__(patience=patience, verbose=verbose, delta=xa.es_delta)

    E.EarlyStopping = FastEarlyStopping

if xa.const_gate:
    # Uniform band weights everywhere: leaf softmax -> 1/K, internal L/R competition -> 0.5/0.5.
    def _const_forward(self, feats):  # feats: [B,C,K,F]
        B, C, K, _ = feats.shape
        return torch.full((B, C, K), 1.0 / K, dtype=feats.dtype, device=feats.device)

    base.BandSelectorMLP.forward = _const_forward
    print("[PRISM-Linear v2] const_gate=1: BandSelectorMLP returns uniform 1/K weights")

_state = {"n": 0, "best": float("inf"), "best_sd": None}
_opts = []

if xa.lr_halve:
    _Adam = torch.optim.Adam

    class _RegAdam(_Adam):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            _opts.append(self)

    torch.optim.Adam = _RegAdam

if xa.lr_halve or xa.load_best:
    _orig_vali = E.Exp_Long_Term_Forecast.vali

    def _vali(self, loader):
        r = _orig_vali(self, loader)
        _state["n"] += 1
        val = r["val_loss"] if isinstance(r, dict) and "val_loss" in r else (
            float(sum(r.values()) / len(r)) if isinstance(r, dict) else float(r))
        if xa.load_best and val < _state["best"]:
            _state["best"] = val
            _state["best_sd"] = copy.deepcopy(self.model.state_dict())
            print(f"[PRISM-Linear v2] new best val {val:.6f} at vali call {_state['n']} (weights kept in memory)")
        if xa.lr_halve:
            for o in _opts:
                for g in o.param_groups:
                    g["lr"] = g.setdefault("initial_lr", g["lr"]) * (0.5 ** _state["n"])
            if _opts:
                print(f"[PRISM-Linear v2] lr after vali call {_state['n']}: {_opts[0].param_groups[0]['lr']:.3e}")
        return r

    E.Exp_Long_Term_Forecast.vali = _vali

if xa.load_best:
    _orig_test = E.Exp_Long_Term_Forecast.test

    def _test(self, *a, **k):
        if _state["best_sd"] is not None:
            self.model.load_state_dict(_state["best_sd"])
            print(f"[PRISM-Linear v2] loaded best-val weights (val {_state['best']:.6f}) before test")
        else:
            print("[PRISM-Linear v2] WARNING: no best state recorded, testing last epoch")
        return _orig_test(self, *a, **k)

    E.Exp_Long_Term_Forecast.test = _test

if __name__ == "__main__":
    args = R.parse_args()
    print(f"[PRISM-Linear v2 2026-11-09] tag={TAG} linear_heads={xa.linear_heads} no_se={xa.no_se} "
          f"const_gate={xa.const_gate} lr_halve={xa.lr_halve} load_best={xa.load_best}")
    print(f"[PRISM-Linear v2] run_dir={run_dir_from_args(args)}")
    if args.use_multi_gpu and len(args.devices.split(",")) > 1:
        world_size = len(args.devices.split(","))
        mp.spawn(R.main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        R.main(0, 1, args)
