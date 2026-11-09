#!/usr/bin/env python3
"""GPU efficiency benchmark for PRISM and the forecasting baselines (rebuttal).

Same setting for every model: ETTh1 (7 channels), seq_len=336, pred_len=96,
batch size 8, fp32, eval mode, no_grad.  For every model we report

  * parameter count
  * FLOPs of one forward pass of the batch (torch.profiler with_flops=True,
    summed over all profiler events; identical to bench_prism_latency.py).
    torch.profiler has no FLOP formula for aten::conv1d (it returns 0 for it,
    verified on torch 2.7), so SCINet / xPatch would be under-counted; we
    therefore also report an analytic conv1d count (forward hooks on every
    nn.Conv1d: 2 * B * C_out * L_out * C_in/groups * k) and the sum of the two.
  * inference latency in ms/batch (CUDA-synchronised, 50 warmup + 200 timed
    iterations, mean and std)
  * inference peak GPU memory (torch.cuda.max_memory_allocated after reset)
  * training-step time (forward + backward + Adam step, 20 warmup + 50 timed)
  * training peak GPU memory

Each model is built from its own code base with the hyper-parameters the PRISM
paper used for the ETTh1 / ctx-336 / H=96 benchmark (see
external/baseline_hparams_main6_compact.md and the per-repo scripts).  Because
the repos share top-level module names (layers, utils, models, model,
data_provider), every model is measured in its own subprocess: the driver
re-invokes this file with ``--only <model>``, the child inserts only that
repo on sys.path, measures, and writes a JSON blob that the driver collects
into a CSV and a LaTeX table.

Usage (driver):
    python bench_efficiency_all.py --device cuda --out_dir results/rebuttal_efficiency
CPU smoke test:
    python bench_efficiency_all.py --device cpu --warmup 2 --iters 3 \
        --train_warmup 1 --train_iters 2 --out_dir /tmp/x
"""
import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "external"
PYDEPS = EXT / "rebuttal_ctx336" / "pydeps"  # provides `ptwt` for WDformer

# ----------------------------------------------------------------------------
# Shared benchmark setting
# ----------------------------------------------------------------------------
SEQ_LEN = 336
PRED_LEN = 96
LABEL_LEN = 48
N_CH = 7
N_MARK = 4            # timeF features for hourly data (ETTh1)
BATCH = 8

# ----------------------------------------------------------------------------
# Model registry.  Each entry: name -> (builder, call_kind, config dict, notes)
#   builder(device) -> torch.nn.Module (eval mode, fp32, on device)
#   call_kind: "x"      -> model(x)
#              "tslib"  -> model(x, x_mark, x_dec, x_mark_dec)
# ----------------------------------------------------------------------------


def _add_path(*parts):
    p = str(Path(*parts))
    if p not in sys.path:
        sys.path.insert(0, p)


def build_prism(tree_depth):
    def _b(device):
        _add_path(REPO)
        from src.arch.prism import prism as PrismModel
        m = PrismModel(
            input_dim=N_CH, hidden_dim=336, num_components=5, dropout=0.25,
            seq_len=SEQ_LEN, pred_len=PRED_LEN, overlap=8, decomp_kind="haar",
            tree_depth=tree_depth, use_last_layer_only=True,
        )
        return m
    return _b


def build_dpad(device):
    _add_path(EXT, "D-PAD")
    from model.D_PAD_SEBlock import DPAD_SE
    return DPAD_SE(
        output_len=PRED_LEN, input_len=SEQ_LEN, input_dim=N_CH,
        enc_hidden=336, dec_hidden=336, num_levels=2, dropout=0.5,
        K_IMP=6, RIN=1,
    )


def build_dlinear(individual):
    def _b(device):
        _add_path(EXT, "DLinear")
        from models.DLinear import Model
        cfg = SimpleNamespace(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=N_CH,
                              individual=individual)
        return Model(cfg)
    return _b


def build_timemixer(device):
    _add_path(EXT, "TimeMixer")
    from models.TimeMixer import Model
    cfg = SimpleNamespace(
        task_name="long_term_forecast", seq_len=SEQ_LEN, label_len=0,
        pred_len=PRED_LEN, enc_in=N_CH, c_out=N_CH, d_model=16, d_ff=32,
        e_layers=2, dropout=0.1, embed="timeF", freq="h", moving_avg=25,
        channel_independence=1, decomp_method="moving_avg", use_norm=1,
        down_sampling_layers=3, down_sampling_window=2,
        down_sampling_method="avg", use_future_temporal_feature=0, top_k=5,
        num_class=1,
    )
    return Model(cfg)


def build_wpmixer(device):
    _add_path(EXT, "WPMixer")   # also provides the vendored pytorch_wavelets
    from models.model import WPMixer
    # ETTh1 / H=96 entry of scripts/Unified/ETTh1_Unified_setup.sh
    return WPMixer(
        c_in=N_CH, c_out=N_CH, seq_len=SEQ_LEN, out_len=PRED_LEN, d_model=256,
        dropout=0.1, embedding_dropout=0.2, device=device, batch_size=128,
        tfactor=3, dfactor=3, wavelet="coif4", level=1, patch_len=16, stride=8,
        no_decomposition=False, use_amp=False,
    )


def build_xpatch(device):
    _add_path(EXT, "xPatch")
    from models.xPatch import Model
    cfg = SimpleNamespace(
        seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=N_CH, patch_len=16, stride=8,
        padding_patch="end", revin=1, ma_type="ema", alpha=0.3, beta=0.3,
    )
    return Model(cfg)


def build_itransformer(device):
    _add_path(EXT, "Time-Series-Library")
    from models.iTransformer import Model
    # TSLib long_term_forecast ETT script: e_layers 2, d_model 128, d_ff 128,
    # factor 3, n_heads 8 (default), dropout 0.1 (default)
    cfg = SimpleNamespace(
        task_name="long_term_forecast", seq_len=SEQ_LEN, pred_len=PRED_LEN,
        enc_in=N_CH, d_model=128, d_ff=128, e_layers=2, n_heads=8, factor=3,
        dropout=0.1, embed="timeF", freq="h", activation="gelu", num_class=1,
    )
    return Model(cfg)


def build_patchtst(device):
    _add_path(EXT, "Time-Series-Library")
    from models.PatchTST import Model
    # TSLib PatchTST_ETTh1.sh: e_layers 1, n_heads 2, factor 3; defaults
    # d_model 512, d_ff 2048, dropout 0.1; patch_len 16, stride 8
    cfg = SimpleNamespace(
        task_name="long_term_forecast", seq_len=SEQ_LEN, pred_len=PRED_LEN,
        enc_in=N_CH, d_model=512, d_ff=2048, e_layers=1, n_heads=2, factor=3,
        dropout=0.1, activation="gelu", num_class=1,
    )
    return Model(cfg, patch_len=16, stride=8)


def build_scinet(device):
    _add_path(EXT, "SCINet")
    from models.SCINet import SCINet
    # scinet_prism6_seeds.sh ETTh1 H=96: hidden-size 4, levels 3, stacks 1,
    # dropout 0.5; run_ETTh.py defaults kernel 5, groups 1, decoder layers 1,
    # concat_len 0, positionalEcoding False, RIN False
    return SCINet(
        output_len=PRED_LEN, input_len=SEQ_LEN, input_dim=N_CH, hid_size=4,
        num_stacks=1, num_levels=3, num_decoder_layer=1, concat_len=0, groups=1,
        kernel=5, dropout=0.5, single_step_output_One=0, positionalE=False,
        modified=True, RIN=False,
    )


def build_paifilter(device):
    _add_path(EXT, "rebuttal_ctx336", "FilterNet")
    from models.PaiFilter import Model
    cfg = SimpleNamespace(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=N_CH,
                          hidden_size=256, embed_size=128)
    return Model(cfg)


def build_wdformer(device):
    _add_path(PYDEPS)
    _add_path(EXT, "rebuttal_ctx336", "WDformer")
    from model.WDformer import Model
    # scripts/ETT/WDformer_ETTh1.sh pred 96: e_layers 2, d_model 256, d_ff 256;
    # run.py defaults wave_size 4, n_heads 8, factor 1, dropout 0.1
    cfg = SimpleNamespace(
        seq_len=SEQ_LEN, pred_len=PRED_LEN, output_attention=False, use_norm=1,
        wave_size=4, d_model=256, d_ff=256, e_layers=2, n_heads=8, factor=1,
        dropout=0.1, embed="timeF", freq="h", activation="gelu",
        class_strategy="projection",
    )
    return Model(cfg)


MODELS = {
    "PRISM_depth2": (build_prism(2), "x",
                     "tree_depth=2, hidden 336, K=5, overlap 8, haar, last_layer_only",
                     "real ETTh1 config (bench_prism_latency.py)"),
    "PRISM_depth1": (build_prism(1), "x",
                     "tree_depth=1, hidden 336, K=5, overlap 8, haar, last_layer_only",
                     ""),
    "D-PAD": (build_dpad, "x",
              "DPAD_SE, enc/dec hidden 336, levels 2, K_IMP 6, dropout 0.5, RIN 1",
              "scripts/run_ETTh1.sh"),
    "DLinear": (build_dlinear(False), "x",
                "shared linear (individual=False), kernel 25",
                "released etth1.sh has no --individual"),
    "DLinear-I": (build_dlinear(True), "x",
                  "per-channel linear (individual=True), kernel 25",
                  "extra row, not the paper setting"),
    "TimeMixer": (build_timemixer, "tslib",
                  "e_layers 2, d_model 16, d_ff 32, down_sampling 3x2 avg, CI=1",
                  "ETT_script/TimeMixer_ETTh1_unify.sh"),
    "WPMixer": (build_wpmixer, "x",
                "d_model 256, coif4 level 1, tfactor 3, dfactor 3, patch 16/8",
                "scripts/Unified/ETTh1_Unified_setup.sh H=96"),
    "xPatch": (build_xpatch, "x",
               "patch 16/8, ema alpha 0.3 beta 0.3, revin 1",
               "scripts/xPatch_fair.sh"),
    "iTransformer": (build_itransformer, "tslib",
                     "e_layers 2, d_model 128, d_ff 128, n_heads 8, factor 3",
                     "TSLib long_term_forecast ETT script"),
    "PatchTST": (build_patchtst, "tslib",
                 "e_layers 1, n_heads 2, d_model 512, d_ff 2048, patch 16/8",
                 "TSLib PatchTST_ETTh1.sh"),
    "SCINet": (build_scinet, "x",
               "hidden 4, levels 3, stacks 1, kernel 5, dropout 0.5",
               "scinet_prism6_seeds.sh ETTh1 H=96; official exp trains float64, measured fp32 here"),
    "PaiFilter": (build_paifilter, "tslib",
                  "hidden_size 256, embed_size=seq_len, RevIN",
                  "FilterNet/models/PaiFilter.py"),
    "WDformer": (build_wdformer, "tslib",
                 "e_layers 2, d_model 256, d_ff 256, wave_size 4, n_heads 8",
                 "scripts/ETT/WDformer_ETTh1.sh pred 96; needs ptwt from pydeps"),
}


# ----------------------------------------------------------------------------
# Measurement helpers (same methodology as bench_prism_latency.py)
# ----------------------------------------------------------------------------

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def make_inputs(device, batch):
    import torch
    x = torch.randn(batch, SEQ_LEN, N_CH, device=device)
    x_mark = torch.zeros(batch, SEQ_LEN, N_MARK, device=device)
    x_dec = torch.zeros(batch, LABEL_LEN + PRED_LEN, N_CH, device=device)
    x_mark_dec = torch.zeros(batch, LABEL_LEN + PRED_LEN, N_MARK, device=device)
    y = torch.randn(batch, PRED_LEN, N_CH, device=device)
    return x, x_mark, x_dec, x_mark_dec, y


def forward(model, kind, inputs):
    x, x_mark, x_dec, x_mark_dec, _ = inputs
    if kind == "tslib":
        out = model(x, x_mark, x_dec, x_mark_dec)
    else:
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def sync(device):
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize()


def measure_flops(model, kind, inputs, device):
    import torch
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.no_grad():
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            forward(model, kind, inputs)
    return float(sum(evt.flops for evt in prof.key_averages() if evt.flops is not None))


def measure_conv1d_flops(model, kind, inputs):
    """Analytic FLOPs of all nn.Conv1d modules (profiler counts them as 0)."""
    import torch
    total = [0.0]

    def hook(mod, inp, out):
        cin_g = mod.in_channels // mod.groups
        total[0] += 2.0 * out.numel() * cin_g * mod.kernel_size[0]

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, torch.nn.Conv1d)]
    with torch.no_grad():
        forward(model, kind, inputs)
    for h in handles:
        h.remove()
    return total[0], len(handles)


def measure_latency(model, kind, inputs, device, warmup, iters):
    import torch
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            forward(model, kind, inputs)
        sync(device)
        for _ in range(iters):
            sync(device)
            t0 = time.perf_counter()
            forward(model, kind, inputs)
            sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std


def measure_infer_memory(model, kind, inputs, device):
    import torch
    if device.type != "cuda":
        return float("nan")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        forward(model, kind, inputs)
    sync(device)
    return torch.cuda.max_memory_allocated(device) / 2 ** 20


def measure_training(model, kind, inputs, device, warmup, iters, lr=1e-4):
    """forward + backward + Adam step; returns (mean_ms, std_ms, peak_mem_MB)."""
    import torch
    y = inputs[-1]
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.MSELoss()

    def step():
        opt.zero_grad(set_to_none=True)
        out = forward(model, kind, inputs)
        loss = crit(out[:, -PRED_LEN:, :], y)
        loss.backward()
        opt.step()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        step()
    sync(device)
    times = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        step()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else float("nan")
    model.eval()
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std, peak


# ----------------------------------------------------------------------------
# Child: measure one model
# ----------------------------------------------------------------------------

def run_one(name, args):
    import torch
    torch.manual_seed(0)
    device = torch.device(args.device)
    builder, kind, config, notes = MODELS[name]
    rec = dict(model=name, status="ok", error="", config=config, notes=notes,
               device=str(device), torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
               batch=args.batch_size, seq_len=SEQ_LEN, pred_len=PRED_LEN, channels=N_CH,
               warmup=args.warmup, iters=args.iters,
               train_warmup=args.train_warmup, train_iters=args.train_iters)
    try:
        model = builder(device).float().to(device).eval()
        rec["params"] = count_params(model)
        rec["params_M"] = rec["params"] / 1e6
        inputs = make_inputs(device, args.batch_size)
        with torch.no_grad():
            out = forward(model, kind, inputs)
        rec["out_shape"] = list(out.shape)
        if tuple(out.shape[-2:]) != (PRED_LEN, N_CH):
            rec["notes"] += f" | WARNING output shape {list(out.shape)}"
        rec["flops_batch"] = measure_flops(model, kind, inputs, device)
        rec["gflops_batch"] = rec["flops_batch"] / 1e9
        rec["gflops_sample"] = rec["flops_batch"] / 1e9 / args.batch_size
        rec["conv1d_flops_batch"], rec["n_conv1d"] = measure_conv1d_flops(model, kind, inputs)
        rec["gflops_batch_total"] = (rec["flops_batch"] + rec["conv1d_flops_batch"]) / 1e9
        rec["latency_ms_mean"], rec["latency_ms_std"] = measure_latency(
            model, kind, inputs, device, args.warmup, args.iters)
        rec["infer_peak_mem_MB"] = measure_infer_memory(model, kind, inputs, device)
        if not args.skip_train:
            try:
                (rec["train_step_ms_mean"], rec["train_step_ms_std"],
                 rec["train_peak_mem_MB"]) = measure_training(
                    model, kind, inputs, device, args.train_warmup, args.train_iters)
            except Exception as e:  # training failure must not void inference numbers
                rec["train_error"] = f"{type(e).__name__}: {e}"
                rec["notes"] += " | training step failed"
                print(traceback.format_exc(), file=sys.stderr)
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"
        print(traceback.format_exc(), file=sys.stderr)
    return rec


# ----------------------------------------------------------------------------
# Driver: one subprocess per model, then CSV + LaTeX
# ----------------------------------------------------------------------------

CSV_COLS = ["model", "status", "error", "params", "params_M", "flops_batch",
            "gflops_batch", "gflops_sample", "conv1d_flops_batch", "n_conv1d",
            "gflops_batch_total", "latency_ms_mean", "latency_ms_std",
            "infer_peak_mem_MB", "train_step_ms_mean", "train_step_ms_std",
            "train_peak_mem_MB", "train_error", "out_shape", "config", "notes",
            "device", "gpu", "torch", "batch", "seq_len", "pred_len", "channels",
            "warmup", "iters", "train_warmup", "train_iters"]


def run_child(name, args, out_dir):
    json_path = out_dir / "json" / f"{name}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--only", name,
           "--json_out", str(json_path), "--device", args.device,
           "--batch_size", str(args.batch_size), "--warmup", str(args.warmup),
           "--iters", str(args.iters), "--train_warmup", str(args.train_warmup),
           "--train_iters", str(args.train_iters)]
    if args.skip_train:
        cmd.append("--skip_train")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PYDEPS) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # never write .pyc into external/*
    log_path = out_dir / "json" / f"{name}.log"
    t0 = time.time()
    with open(log_path, "w") as lf:
        try:
            proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                  timeout=args.timeout, cwd=str(REPO))
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    if json_path.exists():
        rec = json.loads(json_path.read_text())
    else:
        tail = log_path.read_text()[-2000:] if log_path.exists() else ""
        rec = dict(model=name, status="failed",
                   error=f"subprocess rc={rc}; see {log_path}",
                   config=MODELS[name][2], notes=MODELS[name][3] + " | " + tail.strip().splitlines()[-1] if tail.strip() else MODELS[name][3])
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def fmt(v, nd=2):
    if v is None or v == "" or (isinstance(v, float) and v != v):
        return "--"
    return f"{v:.{nd}f}"


def write_latex(records, path, args):
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Model & Params (M) & GFLOPs/batch & GFLOPs/batch (+conv1d) & Latency (ms) & Inf. mem (MB) & Train step (ms) & Train mem (MB) \\\\",
        "\\midrule",
    ]
    for r in records:
        name = r["model"].replace("_", "\\_")
        if r.get("status") != "ok":
            lines.append(f"{name} & \\multicolumn{{7}}{{l}}{{failed: {r.get('error','')[:60].replace('_', chr(92)+'_').replace('&', chr(92)+'&')}}} \\\\")
            continue
        lat = f"{fmt(r.get('latency_ms_mean'), 3)} $\\pm$ {fmt(r.get('latency_ms_std'), 3)}"
        tr = ("--" if r.get("train_step_ms_mean") is None else
              f"{fmt(r.get('train_step_ms_mean'), 2)} $\\pm$ {fmt(r.get('train_step_ms_std'), 2)}")
        lines.append(" & ".join([
            name, fmt(r.get("params_M"), 3), fmt(r.get("gflops_batch"), 3),
            fmt(r.get("gflops_batch_total"), 3), lat,
            fmt(r.get("infer_peak_mem_MB"), 1), tr, fmt(r.get("train_peak_mem_MB"), 1),
        ]) + " \\\\")
    gpu = next((r.get("gpu", "") for r in records if r.get("status") == "ok"), "")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Efficiency on ETTh1 (7 channels, context 336, horizon 96, batch {args.batch_size}, fp32, {gpu}). "
        f"FLOPs: torch.profiler (with\\_flops) for one forward pass of the batch; the +conv1d column adds an analytic count "
        f"for nn.Conv1d, which torch.profiler does not count. Latency: mean $\\pm$ std over {args.iters} "
        f"CUDA-synchronised iterations after {args.warmup} warmup, eval mode, no\\_grad. Train step: forward+backward+Adam, "
        f"{args.train_iters} iterations after {args.train_warmup} warmup. Memory: torch.cuda.max\\_memory\\_allocated.}}",
        "\\label{tab:efficiency_all}",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--batch_size", type=int, default=BATCH)
    ap.add_argument("--warmup", type=int, default=50, help="inference warmup iterations")
    ap.add_argument("--iters", type=int, default=200, help="inference timed iterations")
    ap.add_argument("--train_warmup", type=int, default=20)
    ap.add_argument("--train_iters", type=int, default=50)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of model names (default: all): " + ", ".join(MODELS))
    ap.add_argument("--out_dir", default=str(REPO / "results" / "rebuttal_efficiency"))
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per model subprocess")
    ap.add_argument("--only", default=None, help="(internal) measure a single model in-process")
    ap.add_argument("--json_out", default=None, help="(internal) where --only writes its record")
    args = ap.parse_args()

    if args.only:
        if args.only not in MODELS:
            sys.exit(f"unknown model {args.only}; choose from {list(MODELS)}")
        rec = run_one(args.only, args)
        print(json.dumps(rec, indent=1))
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(rec, indent=1))
        return

    names = args.models or list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        sys.exit(f"unknown models {unknown}; choose from {list(MODELS)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={args.device} batch={args.batch_size} warmup={args.warmup} iters={args.iters} "
          f"train {args.train_warmup}+{args.train_iters}; out_dir={out_dir}", flush=True)

    records = []
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        rec = run_child(name, args, out_dir)
        records.append(rec)
        if rec.get("status") == "ok":
            print(f"  params {rec['params']:,} ({rec['params_M']:.4f} M) | FLOPs/batch {rec['gflops_batch']:.4f} G (+conv1d {rec['gflops_batch_total']:.4f} G) | "
                  f"latency {rec['latency_ms_mean']:.3f} +/- {rec['latency_ms_std']:.3f} ms | "
                  f"inf mem {fmt(rec.get('infer_peak_mem_MB'), 1)} MB | "
                  f"train step {fmt(rec.get('train_step_ms_mean'), 2)} ms | train mem {fmt(rec.get('train_peak_mem_MB'), 1)} MB",
                  flush=True)
        else:
            print(f"  FAILED: {rec.get('error')}", flush=True)

    csv_path = out_dir / "efficiency_all.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS + ["wall_s"], extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    tex_path = out_dir / "efficiency_all.tex"
    write_latex(records, tex_path, args)
    (out_dir / "efficiency_all.json").write_text(json.dumps(records, indent=1))
    print(f"\nwrote {csv_path}\nwrote {tex_path}", flush=True)


if __name__ == "__main__":
    main()
