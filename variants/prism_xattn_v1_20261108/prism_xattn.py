"""PRISM-X v1 (2026-11-08): PRISM + SCRYER-style cross-channel context.

Built on a frozen copy of src/arch/prism.py (prism_base_20261108.py); the
original model file is not modified. Everything new is in this file.

What is added (mirrors ScryerFM / scryer.py, src/arch/layers/selector.py):
  ChannelContext   per-channel embedding from the 6 band statistics of the
                   root-level Haar decomposition (mean over K bands -> LayerNorm
                   -> Linear -> SquaredGELU) + learned channel-identity
                   embedding, then ChannelSelfAttention (pre-LN MHA + FFN) and a
                   second cross-attention block with LN residual
                   (= SCRYER enable_channel_cross_attn).
  router path (alpha)  the context (projected to xattn_emb dims) is concatenated
                   to the 6 band statistics at every router, so band and
                   sibling selection can depend on the other channels. The new
                   input columns are zero-initialised.
  FiLM path        y = y * (1 + alpha) + beta in normalised space, alpha/beta =
                   MLP(context) -> [B, C, H], last layer zero-initialised.
With use_channel_attn=False the model is numerically the frozen PRISM copy.
"""
import math
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import prism_base_20261108 as base  # frozen copy of src/arch/prism.py

# module-level config read by the subclasses while the base constructors build
# the tree (the base code instantiates children by the module-global name
# `TimeTree`, which we rebind to TimeTreeX below).
XCFG = dict(
    use_channel_attn=True,
    xattn_mode="full",  # full | router_only | film_only
    d_model=128,
    emb_dim=16,
    n_heads=4,
    dropout=0.1,
    use_identity=True,
    film_hidden=128,
)


class SquaredGELU(nn.Module):
    def forward(self, x):
        return x * nn.functional.gelu(x)


class BandSelectorMLPX(nn.Module):
    """Router: [6 band stats (+ channel context)] -> softmax over K bands."""

    def __init__(self, F=6, hidden=32, temperature=0.2, emb_dim=0):
        super().__init__()
        self.temperature = temperature
        self.F, self.emb_dim = F, emb_dim
        self.net = nn.Sequential(nn.Linear(F + emb_dim, hidden), nn.LeakyReLU(), nn.Linear(hidden, 1))
        if emb_dim > 0:
            with torch.no_grad():
                self.net[0].weight[:, F:].zero_()  # start identical to PRISM

    def forward(self, feats, channel_emb=None):  # feats [B,C,K,F], channel_emb [B,C,E]
        B, C, K, F = feats.shape
        x = feats
        if self.emb_dim > 0:
            if channel_emb is None:
                ce = feats.new_zeros(B, C, K, self.emb_dim)
            else:
                ce = channel_emb.unsqueeze(2).expand(-1, -1, K, -1).to(feats.dtype)
            x = torch.cat([feats, ce], dim=-1)
        logits = self.net(x.reshape(B * C * K, -1)).view(B, C, K)
        w = torch.softmax(logits / self.temperature, dim=2)
        return w / w.sum(dim=2, keepdim=True)


class ChannelSelfAttention(nn.Module):
    """Pre-LN multi-head self-attention over channels + FFN (as in SCRYER)."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), SquaredGELU(), nn.Dropout(dropout),
                                nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x):  # [B,C,d]
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class ChannelContext(nn.Module):
    def __init__(self, C, feat_dim=6, d_model=128, n_heads=4, dropout=0.1, use_identity=True):
        super().__init__()
        self.embed = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d_model), SquaredGELU())
        self.id_emb = nn.Embedding(C, d_model) if use_identity else None
        if self.id_emb is not None:
            nn.init.normal_(self.id_emb.weight, std=0.02)
        self.self_attn = ChannelSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_ln = nn.LayerNorm(d_model)

    def forward(self, feats):  # feats [B,C,K,6] -> [B,C,d]
        B, C = feats.shape[:2]
        h = self.embed(feats.mean(dim=2).float())
        if self.id_emb is not None:
            h = h + self.id_emb(torch.arange(C, device=feats.device)).unsqueeze(0)
        h = self.self_attn(h)
        ce = self.cross_ln(h + self.cross_attn(h, h, h, need_weights=False)[0])
        return h + ce


class TimeTreeX(base.TimeTree):
    """TimeTree whose routers also see the channel context (alpha path)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        emb = XCFG["emb_dim"] if XCFG["use_channel_attn"] and XCFG["xattn_mode"] in ("full", "router_only") else 0
        if hasattr(self, "mlp_score"):
            old = self.mlp_score
            self.mlp_score = BandSelectorMLPX(F=6, hidden=old.net[0].out_features,
                                              temperature=old.temperature, emb_dim=emb)

    def forward(self, x, mode="route", meta=None, side=None, global_step=0, y=None, stats=None, channel_emb=None):
        if self.depth == 0:
            imfs_leaf = self._decompose(x)
            w = self.mlp_score(self.band_feat(imfs_leaf), channel_emb)
            return x, {"level0": imfs_leaf * w.unsqueeze(-1)}
        xL_raw, xR_raw, meta = self._split_with_overlap(x)
        imfsL, imfsR = self._decompose(xL_raw), self._decompose(xR_raw)
        wL_raw = self.mlp_score(self.band_feat(imfsL), channel_emb)
        wR_raw = self.mlp_score(self.band_feat(imfsR), channel_emb)
        comp = torch.softmax(torch.stack([wL_raw, wR_raw], dim=-1) / self.temperature, dim=-1)
        wL, wR = comp[..., 0], comp[..., 1]
        self._current_wL, self._current_wR = wL, wR
        self._last_wL, self._last_wR = wL, wR
        wbL, wbR = imfsL * wL.unsqueeze(-1), imfsR * wR.unsqueeze(-1)
        yL, dictL = self.left(wbL.sum(dim=2), mode=mode, meta=meta, side="L", global_step=global_step,
                              y=y, stats=stats, channel_emb=channel_emb)
        yR, dictR = self.right(wbR.sum(dim=2), mode=mode, meta=meta, side="R", global_step=global_step,
                               y=y, stats=stats, channel_emb=channel_emb)
        out = self._stitch(yL, yR, meta)
        bands_dict = {}
        for key in set(dictL) | set(dictR):
            if key in dictL and key in dictR:
                bands_dict[key] = torch.cat([dictL[key], dictR[key]], dim=2)
            else:
                bands_dict[key] = dictL.get(key, dictR.get(key))
        bands_dict[f"level{self.total_depth - self.depth}"] = torch.cat([wbL, wbR], dim=2)
        if mode == "forecast" and self.depth == self.total_depth:
            return self.forecast_all_levels(bands_dict), None
        return self.mix(out), bands_dict


class TimeSplitBackboneX(base.TimeSplitBackbone):
    def __init__(self, input_dim, *args, **kwargs):
        super().__init__(input_dim, *args, **kwargs)
        self.use_channel_attn = bool(XCFG["use_channel_attn"])
        self.xattn_mode = XCFG["xattn_mode"]
        if self.use_channel_attn:
            d = XCFG["d_model"]
            self.band_feat_root = base.IMFSharpFeatTiny()
            self.channel_ctx = ChannelContext(input_dim, 6, d, XCFG["n_heads"], XCFG["dropout"], XCFG["use_identity"])
            self.to_router = nn.Linear(d, XCFG["emb_dim"]) if self.xattn_mode in ("full", "router_only") else None
            if self.xattn_mode in ("full", "film_only"):
                H, fh = self.tree.pred_len, XCFG["film_hidden"]
                self.film_alpha = nn.Sequential(nn.Linear(d, fh), SquaredGELU(), nn.Linear(fh, H))
                self.film_beta = nn.Sequential(nn.Linear(d, fh), SquaredGELU(), nn.Linear(fh, H))
                for m in (self.film_alpha[-1], self.film_beta[-1]):
                    nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
            else:
                self.film_alpha = self.film_beta = None
        self.last_context = None

    def forward(self, x, t=None, features=None, mode="route", y=None):  # x [B,L,C]
        x_c = x.transpose(1, 2).contiguous()
        x_in, stats = (self.norm(x_c) if self.norm else (x_c, None))
        channel_emb, h = None, None
        if self.use_channel_attn:
            feats = self.band_feat_root(self.tree._decompose(x_in))  # [B,C,K,6], no grad
            h = self.channel_ctx(feats)                              # [B,C,d]
            self.last_context = h.detach()
            if self.to_router is not None:
                channel_emb = self.to_router(h)
        y_in = None
        if y is not None:
            y_c = y.transpose(1, 2).contiguous()
            if self.norm and stats is not None:
                mu, sig = stats
                y_in = ((y_c - mu) / sig).transpose(1, 2)
            else:
                y_in = y_c.transpose(1, 2)
        out, _ = self.tree(x_in, y=y_in, mode=mode, channel_emb=channel_emb)
        if mode == "forecast":
            if self.use_channel_attn and self.film_alpha is not None:
                out = out * (1.0 + self.film_alpha(h).to(out.dtype)) + self.film_beta(h).to(out.dtype)
            if self.norm:
                out = self.norm.invert(out, stats)
            return out.transpose(1, 2).contiguous()
        if self.norm:
            out = self.norm.invert(out, stats)
        return out.transpose(1, 2).contiguous()


# rebind the names the base constructors use, so the tree is built from the X classes
base.TimeTree = TimeTreeX
base.TimeSplitBackbone = TimeSplitBackboneX


class prism(base.prism):
    """Drop-in replacement for src.arch.prism.prism with extra xattn kwargs."""

    def __init__(self, *args, use_channel_attn=True, xattn_mode="full", xattn_d=128, xattn_emb=16,
                 xattn_heads=4, xattn_dropout=0.1, xattn_identity=True, **kwargs):
        XCFG.update(use_channel_attn=use_channel_attn, xattn_mode=xattn_mode, d_model=xattn_d, emb_dim=xattn_emb,
                    n_heads=xattn_heads, dropout=xattn_dropout, use_identity=xattn_identity)
        super().__init__(*args, **kwargs)


if __name__ == "__main__":  # CPU smoke
    torch.manual_seed(0)
    for C, L, H, on, mode in [(7, 336, 96, True, "full"), (862, 336, 96, True, "full"), (7, 336, 96, False, "full"),
                              (21, 336, 720, True, "router_only"), (21, 336, 192, True, "film_only")]:
        m = prism(input_dim=C, hidden_dim=64, num_components=5, dropout=0.1, seq_len=L, pred_len=H, overlap=8,
                  tree_depth=2, save_dir="/tmp/prism_xattn_smoke", use_channel_attn=on, xattn_mode=mode)
        x = torch.randn(4, L, C)
        out = m(x)
        out.pow(2).mean().backward()
        n = sum(p.numel() for p in m.parameters())
        nx = sum(p.numel() for k, p in m.named_parameters() if "channel_ctx" in k or "film" in k or "to_router" in k)
        print(f"C={C} H={H} on={on} mode={mode} out={tuple(out.shape)} params={n} xattn_params={nx} "
              f"finite={bool(torch.isfinite(out).all())}")
    # identity check: with xattn on, zero-init paths -> same output as off at init?
    torch.manual_seed(1); a = prism(input_dim=7, hidden_dim=64, num_components=5, seq_len=336, pred_len=96, overlap=8,
                                    tree_depth=2, save_dir="/tmp/prism_xattn_smoke", use_channel_attn=False)
    torch.manual_seed(1); b = prism(input_dim=7, hidden_dim=64, num_components=5, seq_len=336, pred_len=96, overlap=8,
                                    tree_depth=2, save_dir="/tmp/prism_xattn_smoke", use_channel_attn=True)
    sa = a.state_dict()
    sd = {}
    for k, v in b.state_dict().items():
        if k not in sa:
            continue
        sd[k] = v if v.shape == sa[k].shape else v[:, : sa[k].shape[1]]  # router first layer: keep the 6 stat columns
    missing = a.load_state_dict(sd, strict=False); a.eval(); b.eval()
    print("unmatched base keys:", missing.missing_keys)
    x = torch.randn(3, 336, 7)
    print("max |on-off| at init (expect ~0):", (a(x) - b(x)).abs().max().item())
