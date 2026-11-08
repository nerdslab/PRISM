#!/usr/bin/env python
"""Summarise the exchange grid: per (config, H) best-val epoch, validation MSE/MAE at that epoch, test MSE/MAE.
Selection rule (fixed before looking at test): per horizon pick the config with the lowest validation MSE.
Usage: python select_exchange.py [logs_dir]"""
import glob, os, re, sys
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
rows = []
for f in sorted(glob.glob(os.path.join(root, "exg_*", "exchange_cont336_M", "pred*_cont336_s*.log"))):
    cfg = f.split(os.sep)[-3][4:]; H = int(re.search(r"pred(\d+)_", f).group(1))
    txt = open(f).read()
    vals = [float(x) for x in re.findall(r"^\[epoch \d+\] val_loss=([0-9.]+)", txt, re.M)]
    normed = re.findall(r"\|  Normed  \| mse:([0-9.]+) \| mae:([0-9.]+) \|", txt)
    tm = re.search(r"^test mae:([0-9.e-]+), mse:([0-9.e-]+)", txt, re.M)   # labels swapped in run_prism.py: first is MSE
    if not vals or not tm:
        rows.append((cfg, H, None, None, None, None, None, len(vals))); continue
    b = min(range(len(vals)), key=lambda i: vals[i])
    vmse, vmae = (float(normed[b][0]), float(normed[b][1])) if b < len(normed) else (None, None)
    rows.append((cfg, H, b, vmse, vmae, float(tm.group(1)), float(tm.group(2)), len(vals)))
print(f"{'config':16s} {'H':>4s} {'bestep':>6s} {'val_mse':>8s} {'val_mae':>8s} {'test_mse':>9s} {'test_mae':>9s} {'epochs':>6s}")
for r in rows:
    print(f"{r[0]:16s} {r[1]:4d} " + " ".join("   n/a" if v is None else (f"{v:8.4f}" if isinstance(v, float) else f"{v:6d}") for v in r[2:]))
print("\nDLinear (paper, ctx 336) exchange: 96 0.081/0.203  192 0.157/0.293  336 0.305/0.414  720 0.643/0.601")
print("Selection by lowest validation MSE per horizon:")
for H in (96, 192, 336, 720):
    c = [r for r in rows if r[1] == H and r[3] is not None]
    if c:
        s = min(c, key=lambda r: r[3]); ref = [r for r in c if r[0] == "ref"]
        print(f"  H={H}: {s[0]:14s} test {s[5]:.3f}/{s[6]:.3f}" + (f"   (ref {ref[0][5]:.3f}/{ref[0][6]:.3f})" if ref else ""))
