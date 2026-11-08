"""PRISM-X v2 (2026-11-08, afternoon): WDformer-style channel tokens on top of v1.

Differences from v1 (prism_xattn.py), everything else inherited:
  token      per-channel token = v1's 6 band statistics (optional) + a WDformer-style
             Haar token: orthonormal Haar wavedec (level `wave_level`, default 4) of the
             normalised 336-step history, one Linear per band into a slice of d_model
             (DataEmbedding_wd in WDformer/layers/Embed.py), + channel-identity embedding.
  depth      `n_layers` pre-LN self-attention blocks (v1: one block + one cross block).
  read-out   zero-initialised Linear(d_model -> H) added to the per-channel forecast in
             normalised space, so other channels can contribute *shape*, not only scale.
             The v1 alpha (router) and FiLM paths are kept.
With use_channel_attn=False the model is still the frozen PRISM copy.
"""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import prism_xattn as v1  # noqa: E402  (this also rebinds base.TimeTree -> TimeTreeX)
from prism_xattn import XCFG, SquaredGELU, ChannelSelfAttention, base  # noqa: E402

XCFG.update(token="both", n_layers=2, readout=True, wave_level=4, seq_len=336)


def haar_wavedec(x, level):
    """Orthonormal Haar DWT along the last dim. x [..., L] -> [cA_level, cD_level, ..., cD_1] (ptwt order)."""
    details = []
    a = x
    for _ in range(level):
        if a.shape[-1] % 2:
            a = torch.nn.functional.pad(a, (0, 1), mode="replicate")
        e, o = a[..., 0::2], a[..., 1::2]
        details.append((e - o) * 0.7071067811865476)
        a = (e + o) * 0.7071067811865476
    return [a] + details[::-1]


class HaarTokenEmbed(nn.Module):
    """WDformer DataEmbedding_wd: per-band Linear into slices of d_model, concatenated."""

    def __init__(self, L, level, d_model):
        super().__init__()
        self.level = level
        lens = [c.shape[-1] for c in haar_wavedec(torch.zeros(1, L), level)]
        d_len = d_model // (level + 1)
        dims = [d_len] * level + [d_model - level * d_len]
        self.linears = nn.ModuleList(nn.Linear(n, d) for n, d in zip(lens, dims))

    def forward(self, x):  # x [B,C,L] (normalised history) -> [B,C,d]
        coeffs = haar_wavedec(x.float(), self.level)
        return torch.cat([lin(c) for lin, c in zip(self.linears, coeffs)], dim=-1)


class ChannelContextV2(nn.Module):
    def __init__(self, C, L, feat_dim=6, d_model=256, n_heads=4, dropout=0.1, use_identity=True,
                 token="both", n_layers=2, wave_level=4):
        super().__init__()
        self.token = token
        self.stats_embed = (nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d_model), SquaredGELU())
                            if token in ("stats", "both") else None)
        self.wave_embed = HaarTokenEmbed(L, wave_level, d_model) if token in ("wave", "both") else None
        self.id_emb = nn.Embedding(C, d_model) if use_identity else None
        if self.id_emb is not None:
            nn.init.normal_(self.id_emb.weight, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(ChannelSelfAttention(d_model, n_heads, dropout) for _ in range(n_layers))
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, feats, x_in):  # feats [B,C,K,6], x_in [B,C,L] -> [B,C,d]
        B, C = x_in.shape[:2]
        h = 0
        if self.stats_embed is not None:
            h = h + self.stats_embed(feats.mean(dim=2).float())
        if self.wave_embed is not None:
            h = h + self.wave_embed(x_in)
        if self.id_emb is not None:
            h = h + self.id_emb(torch.arange(C, device=x_in.device)).unsqueeze(0)
        h = self.drop(h)
        for blk in self.blocks:
            h = blk(h)
        return self.ln_out(h)


class TimeSplitBackboneX2(v1.TimeSplitBackboneX):
    def __init__(self, input_dim, *args, **kwargs):
        super().__init__(input_dim, *args, **kwargs)
        if self.use_channel_attn:
            d, L = XCFG["d_model"], kwargs.get("seq_len", XCFG["seq_len"])
            self.channel_ctx = ChannelContextV2(input_dim, L, 6, d, XCFG["n_heads"], XCFG["dropout"],
                                                XCFG["use_identity"], XCFG["token"], XCFG["n_layers"],
                                                XCFG["wave_level"])
            if XCFG["readout"]:
                self.readout = nn.Linear(d, self.tree.pred_len)
                nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)
            else:
                self.readout = None

    def forward(self, x, t=None, features=None, mode="route", y=None):  # x [B,L,C]
        x_c = x.transpose(1, 2).contiguous()
        x_in, stats = (self.norm(x_c) if self.norm else (x_c, None))
        channel_emb, h = None, None
        if self.use_channel_attn:
            feats = self.band_feat_root(self.tree._decompose(x_in))  # [B,C,K,6], no grad
            h = self.channel_ctx(feats, x_in)                        # [B,C,d]
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
            if self.use_channel_attn:
                if self.readout is not None:
                    out = out + self.readout(h).to(out.dtype)
                if self.film_alpha is not None:
                    out = out * (1.0 + self.film_alpha(h).to(out.dtype)) + self.film_beta(h).to(out.dtype)
            if self.norm:
                out = self.norm.invert(out, stats)
            return out.transpose(1, 2).contiguous()
        if self.norm:
            out = self.norm.invert(out, stats)
        return out.transpose(1, 2).contiguous()


base.TimeSplitBackbone = TimeSplitBackboneX2


class prism(v1.prism):
    def __init__(self, *args, xattn_token="both", xattn_layers=2, xattn_readout=True, xattn_wave_level=4, **kwargs):
        XCFG.update(token=xattn_token, n_layers=xattn_layers, readout=xattn_readout, wave_level=xattn_wave_level,
                    seq_len=kwargs.get("seq_len", 336))
        super().__init__(*args, **kwargs)


if __name__ == "__main__":  # CPU smoke
    torch.manual_seed(0)
    # Haar check against a brute-force two-level transform
    x = torch.randn(2, 3, 8)
    c = haar_wavedec(x, 2)
    assert [t.shape[-1] for t in c] == [2, 2, 4], [t.shape[-1] for t in c]
    assert torch.allclose(c[-1][..., 0], (x[..., 0] - x[..., 1]) / 2 ** 0.5)
    assert torch.allclose(sum((t ** 2).sum() for t in c), (x ** 2).sum(), atol=1e-4)  # orthonormal
    for C, L, H, on, tok in [(7, 336, 96, True, "both"), (862, 336, 96, True, "both"), (7, 336, 96, False, "both"),
                             (21, 336, 720, True, "wave"), (21, 336, 192, True, "stats")]:
        m = prism(input_dim=C, hidden_dim=64, num_components=5, dropout=0.1, seq_len=L, pred_len=H, overlap=8,
                  tree_depth=2, save_dir="/tmp/prism_xattn_smoke", use_channel_attn=on, xattn_d=256,
                  xattn_token=tok, xattn_layers=2)
        xx = torch.randn(4, L, C)
        out = m(xx)
        out.pow(2).mean().backward()
        n = sum(p.numel() for p in m.parameters())
        nx = sum(p.numel() for k, p in m.named_parameters()
                 if any(s in k for s in ("channel_ctx", "film", "to_router", "readout")))
        print(f"C={C} H={H} on={on} token={tok} out={tuple(out.shape)} params={n} xattn_params={nx} "
              f"finite={bool(torch.isfinite(out).all())}")
    torch.manual_seed(1); a = prism(input_dim=7, hidden_dim=64, num_components=5, seq_len=336, pred_len=96, overlap=8,
                                    tree_depth=2, save_dir="/tmp/prism_xattn_smoke", use_channel_attn=False)
    torch.manual_seed(1); b = prism(input_dim=7, hidden_dim=64, num_components=5, seq_len=336, pred_len=96, overlap=8,
                                    tree_depth=2, save_dir="/tmp/prism_xattn_smoke", use_channel_attn=True, xattn_d=256)
    sa = a.state_dict(); sd = {}
    for k, v in b.state_dict().items():
        if k in sa:
            sd[k] = v if v.shape == sa[k].shape else v[:, : sa[k].shape[1]]
    a.load_state_dict(sd, strict=False); a.eval(); b.eval()
    xx = torch.randn(3, 336, 7)
    print("max |on-off| at init (expect ~0):", (a(xx) - b(xx)).abs().max().item())
