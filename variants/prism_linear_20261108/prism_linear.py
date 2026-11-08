"""PRISM-Linear (2026-11-08): PRISM with DLinear-style linear forecast heads.

Built on the frozen copy variants/prism_xattn_v1_20261108/prism_base_20261108.py (= src/arch/prism.py as of
2026-11-08); src/ is not touched.  Options (module-level LCFG, set through the `prism` kwargs):
  linear_heads  replace every per-band head  Linear(L_i, hidden) -> LeakyReLU -> Linear(hidden, H)
                by a single Linear(L_i, H)  (what DLinear does per decomposition component)
  no_se         replace the SE block over bands by identity, so the forecast is a routed sum of per-band
                linear maps: y = sum_b w_b(x) * W_b x_b   (w_b are PRISM's input-dependent band weights)
With both False the model is exactly the frozen PRISM copy.
"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, "..", "prism_xattn_v1_20261108")
for p in (V1, os.path.abspath(os.path.join(HERE, "..", ".."))):
    if p not in sys.path:
        sys.path.insert(0, p)
import prism_base_20261108 as base  # noqa: E402

LCFG = dict(linear_heads=True, no_se=True)


class _Id(nn.Identity):
    """Identity that exposes in/out_features (the base tree's print helper reads proj2[0].out_features)."""

    def __init__(self, n):
        super().__init__()
        self.in_features = self.out_features = n


class TimeTreeL(base.TimeTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "activate"):   # child nodes have no heads
            return
        if LCFG["linear_heads"]:
            for level in range(8):
                n1 = f"forecast_proj1_level{level}"
                if not hasattr(self, n1):
                    continue
                old1 = getattr(self, n1)
                setattr(self, n1, nn.ModuleList(nn.Linear(l.in_features, self.pred_len) for l in old1))
                setattr(self, f"forecast_proj2_level{level}", nn.ModuleList(_Id(self.pred_len) for _ in old1))
            self.activate = nn.Identity()
        if LCFG["no_se"]:
            self.SEBlock_all = nn.Identity()


base.TimeTree = TimeTreeL


class prism(base.prism):
    def __init__(self, *args, linear_heads=True, no_se=True, **kwargs):
        LCFG.update(linear_heads=linear_heads, no_se=no_se)
        super().__init__(*args, **kwargs)


if __name__ == "__main__":  # CPU smoke
    torch.manual_seed(0)
    for lh, nse, depth, last in [(True, True, 2, False), (True, False, 2, False), (True, True, 1, False),
                                 (True, True, 2, True), (False, False, 2, False)]:
        m = prism(input_dim=8, hidden_dim=336, num_components=5, dropout=0.0, seq_len=336, pred_len=720, overlap=8,
                  tree_depth=depth, save_dir="/tmp/prism_linear_smoke", use_last_layer_only=last,
                  linear_heads=lh, no_se=nse)
        x = torch.randn(4, 336, 8); y = m(x); y.pow(2).mean().backward()
        n = sum(p.numel() for p in m.parameters())
        print(f"linear={lh} no_se={nse} depth={depth} last={last} out={tuple(y.shape)} params={n} finite={bool(torch.isfinite(y).all())}")
    # linearity check in normalised space is not exact because of SeriesNorm/routing; check head is linear:
    torch.manual_seed(1)
    m = prism(input_dim=8, hidden_dim=336, num_components=5, seq_len=336, pred_len=96, overlap=8, tree_depth=2,
              save_dir="/tmp/prism_linear_smoke", linear_heads=True, no_se=True)
    t = m.backbone.tree
    print("head modules:", type(t.forecast_proj1_level0[0]).__name__, type(t.forecast_proj2_level0[0]).__name__,
          type(t.activate).__name__, type(t.SEBlock_all).__name__)
