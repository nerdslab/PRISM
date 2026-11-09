#!/usr/bin/env python
"""DLinear with / without per-window instance normalisation (2026-11-09).

Runs the released DLinear code in external/DLinear unchanged (its run_longExp.py is executed via runpy), but when
--instnorm 1 is given the model registered under the name 'DLinear' is wrapped as

    mu, sig = mean_t(x), std_t(x)          # per sample, per channel, over the look-back window
    y = DLinear((x - mu) / sig) * sig + mu

which is exactly PRISM's SeriesNorm (eps 1e-5) around DLinear.  Everything else (data split, MSE loss,
lr halving, early stopping on validation, reload of the best checkpoint, metrics) is the original DLinear code.

Usage: python run_dlinear_in.py --instnorm {0,1} <all run_longExp.py arguments>
Checkpoints / result.txt / test_results are written to the current working directory (run from this folder).
"""
import argparse
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.abspath(os.path.join(HERE, "..", "..", "external", "DLinear"))
sys.path.insert(0, DL)

p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
p.add_argument("--instnorm", type=int, default=0)
xa, rest = p.parse_known_args()
sys.argv = [os.path.join(DL, "run_longExp.py")] + rest

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import exp.exp_main as exp_main  # noqa: E402
from models import DLinear as _DLinear  # noqa: E402

if xa.instnorm:
    class _Wrapped(nn.Module):
        def __init__(self, configs):
            super().__init__()
            self.core = _DLinear.Model(configs)
            self.eps = 1e-5

        def forward(self, x):  # x: [B, L, C]
            mu = x.mean(dim=1, keepdim=True)
            sig = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
            y = self.core((x - mu) / sig)
            return y * sig + mu

    class _Mod:  # stands in for the `models.DLinear` module inside exp_main's model_dict
        Model = _Wrapped

    exp_main.DLinear = _Mod
    print("[DLinear-IN] instnorm=1: DLinear wrapped in per-window mean/std normalisation (SeriesNorm)")
else:
    print("[DLinear-IN] instnorm=0: released DLinear unchanged")

runpy.run_path(sys.argv[0], run_name="__main__")
