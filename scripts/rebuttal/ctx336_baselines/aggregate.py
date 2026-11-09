"""Aggregate ctx-336 single-seed re-runs of the dagger baselines and compare with
(a) the published look-back-96 numbers used in the rebuttal's Table 1 and
(b) PRISM's Table 1 numbers (ctx 336, 8 seeds).
Outputs: results/rebuttal_ctx336_baselines/{ctx336_table.csv, summary.md, ctx336_table.tex}
"""
import re, csv, json, os, sys
ROOT = "/vast/projects/dyer1/lab/user/zihao/PRISM"
OUT = f"{ROOT}/results/rebuttal_ctx336_baselines"
DS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "traffic", "electricity", "exchange", "weather"]
DSN = {"ETTh1": "ETTh1", "ETTh2": "ETTh2", "ETTm1": "ETTm1", "ETTm2": "ETTm2", "traffic": "Traffic",
       "electricity": "Electricity", "exchange": "Exchange", "weather": "Weather"}
HS = [96, 192, 336, 720]

def T(s):  # "mse mae; mse mae; ..." per horizon -> list of (mse, mae) or None
    out = []
    for h in s.split(";"):
        h = h.strip()
        out.append(None if h == "-" else tuple(float(v) for v in h.split()))
    return out

# PRISM, Table 1 (ctx 336, 8 seeds), transcribed from the rebuttal's updated Table 1 image
PRISM = {
 "ETTh1": T("0.355 0.374; 0.390 0.405; 0.386 0.407; 0.421 0.445"),
 "ETTh2": T("0.261 0.322; 0.311 0.359; 0.308 0.361; 0.387 0.420"),
 "ETTm1": T("0.282 0.326; 0.323 0.352; 0.354 0.377; 0.423 0.409"),
 "ETTm2": T("0.158 0.234; 0.214 0.267; 0.264 0.319; 0.352 0.372"),
 "traffic": T("0.362 0.237; 0.376 0.244; 0.401 0.257; 0.433 0.271"),
 "electricity": T("0.122 0.219; 0.145 0.235; 0.159 0.243; 0.194 0.281"),
 "exchange": T("0.081 0.198; 0.173 0.297; 0.314 0.396; 0.712 0.606"),
 "weather": T("0.140 0.177; 0.187 0.227; 0.234 0.266; 0.310 0.320"),
}
# published look-back-96 numbers (dagger columns of the updated Table 1)
LIT = {
 "FreqCycle": {
  "ETTh1": T("0.369 0.390; 0.417 0.418; 0.459 0.439; 0.456 0.454"),
  "ETTh2": T("0.282 0.336; 0.361 0.388; 0.415 0.428; 0.425 0.443"),
  "ETTm1": T("0.310 0.353; 0.351 0.374; 0.383 0.396; 0.446 0.434"),
  "ETTm2": T("0.162 0.244; 0.226 0.286; 0.283 0.326; 0.383 0.387"),
  "traffic": T("0.438 0.262; 0.434 0.258; 0.446 0.258; 0.476 0.265"),
  "electricity": T("0.140 0.234; 0.154 0.246; 0.170 0.264; 0.210 0.297"),
  "exchange": T("-; -; -; -"),
  "weather": T("0.159 0.203; 0.208 0.248; 0.264 0.289; 0.343 0.342")},
 "AWEMixer": {
  "ETTh1": T("0.374 0.391; 0.421 0.425; 0.466 0.442; 0.467 0.469"),
  "ETTh2": T("0.282 0.336; 0.361 0.387; 0.395 0.421; 0.411 0.431"),
  "ETTm1": T("0.316 0.354; 0.359 0.381; 0.391 0.402; 0.443 0.438"),
  "ETTm2": T("0.168 0.257; 0.236 0.296; 0.291 0.336; 0.397 0.391"),
  "traffic": T("-; -; -; -"),
  "electricity": T("0.152 0.245; 0.161 0.254; 0.176 0.275; 0.222 0.308"),
  "exchange": T("0.084 0.201; 0.175 0.293; 0.335 0.431; 0.894 0.685"),
  "weather": T("0.164 0.203; 0.199 0.246; 0.250 0.292; 0.331 0.339")},
 "PaiFilter": {
  "ETTh1": T("0.375 0.394; 0.436 0.422; 0.476 0.443; 0.474 0.469"),
  "ETTh2": T("0.292 0.343; 0.369 0.395; 0.420 0.432; 0.430 0.446"),
  "ETTm1": T("0.318 0.358; 0.364 0.383; 0.396 0.406; 0.456 0.444"),
  "ETTm2": T("0.174 0.257; 0.240 0.300; 0.297 0.339; 0.392 0.393"),
  "traffic": T("0.506 0.336; 0.508 0.333; 0.518 0.335; 0.553 0.354"),
  "electricity": T("0.176 0.264; 0.185 0.270; 0.202 0.286; 0.242 0.319"),
  "exchange": T("0.083 0.202; 0.174 0.296; 0.326 0.413; 0.840 0.670"),
  "weather": T("0.164 0.210; 0.214 0.252; 0.268 0.293; 0.344 0.342")},
 "WDformer": {
  "ETTh1": T("0.391 0.409; 0.445 0.438; 0.491 0.466; 0.514 0.496"),
  "ETTh2": T("0.302 0.349; 0.382 0.402; 0.425 0.432; 0.433 0.446"),
  "ETTm1": T("0.339 0.373; 0.384 0.399; 0.422 0.423; 0.489 0.461"),
  "ETTm2": T("0.181 0.263; 0.247 0.305; 0.314 0.350; 0.417 0.408"),
  "traffic": T("0.391 0.265; 0.403 0.271; 0.423 0.280; 0.456 0.298"),
  "electricity": T("0.144 0.237; 0.161 0.253; 0.173 0.267; 0.205 0.296"),
  "exchange": T("0.089 0.210; 0.188 0.310; 0.337 0.422; 0.817 0.685"),
  "weather": T("0.172 0.212; 0.224 0.256; 0.281 0.298; 0.356 0.348")},
}
MODELS = ["PaiFilter", "TexFilter", "FreqCycle", "AWEMixer", "WDformer"]
LABEL = {"PaiFilter": "FilterNet (PaiFilter)", "TexFilter": "FilterNet (TexFilter)", "FreqCycle": "FreqCycle",
         "AWEMixer": "AWEMixer", "WDformer": "WDformer"}

# ---- parse re-run results
RES = {}
for line in open(f"{OUT}/results_raw.txt"):
    m = re.match(r"(\w+?)_(\w+)_sl336_pl(\d+)\s+mse:([\d.]+), mae:([\d.]+)", line.strip())
    if not m: continue
    model, ds, h, mse, mae = m.group(1), m.group(2), int(m.group(3)), float(m.group(4)), float(m.group(5))
    RES[(model, ds, h)] = (mse, mae)
print("parsed", len(RES), "results")

# ---- long csv
rows = []
for model in MODELS:
    for ds in DS:
        for i, h in enumerate(HS):
            r = RES.get((model, ds, h))
            lit_key = "PaiFilter" if model == "TexFilter" else model
            lit = LIT[lit_key][ds][i]
            p = PRISM[ds][i]
            rows.append(dict(model=model, dataset=ds, H=h,
                             mse_336=r[0] if r else None, mae_336=r[1] if r else None,
                             mse_lit96=lit[0] if lit else None, mae_lit96=lit[1] if lit else None,
                             prism_mse=p[0], prism_mae=p[1]))
with open(f"{OUT}/ctx336_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- summaries
def f3(v): return "–" if v is None else f"{v:.3f}"
md = ["# ctx-336 re-runs of dagger baselines (single seed, released hyperparameters)\n"]
md.append("## Per-model summary\n")
md.append("| model | settings run | ctx336 better than lit-96 (MSE) | (MAE) | PRISM better than ctx336 (MSE) | (MAE) | mean MSE lit96 → ctx336 (common settings) |")
md.append("|---|---|---|---|---|---|---|")
summary = {}
for model in MODELS:
    n = 0; b_mse = b_mae = 0; n_lit = 0; p_mse = p_mae = 0; s_lit = s_336 = 0.0
    for ds in DS:
        for i, h in enumerate(HS):
            r = RES.get((model, ds, h))
            if r is None: continue
            n += 1
            p = PRISM[ds][i]
            if round(p[0],3) < round(r[0],3): p_mse += 1
            if round(p[1],3) < round(r[1],3): p_mae += 1
            lit_key = "PaiFilter" if model == "TexFilter" else model
            lit = LIT[lit_key][ds][i]
            if lit:
                n_lit += 1; s_lit += lit[0]; s_336 += r[0]
                if round(r[0],3) < lit[0]: b_mse += 1
                if round(r[1],3) < lit[1]: b_mae += 1
    summary[model] = dict(n=n, n_lit=n_lit, b_mse=b_mse, b_mae=b_mae, p_mse=p_mse, p_mae=p_mae,
                          mean_lit=s_lit/max(n_lit,1), mean_336=s_336/max(n_lit,1))
    md.append(f"| {LABEL[model]} | {n} | {b_mse}/{n_lit} | {b_mae}/{n_lit} | {p_mse}/{n} | {p_mae}/{n} | {s_lit/max(n_lit,1):.3f} → {s_336/max(n_lit,1):.3f} |")
md.append("\n## Settings where a ctx-336 re-run beats PRISM (MSE or MAE)\n")
for model in MODELS:
    for ds in DS:
        for i, h in enumerate(HS):
            r = RES.get((model, ds, h))
            if r is None: continue
            p = PRISM[ds][i]
            if round(r[0],3) < p[0] or round(r[1],3) < p[1]:
                md.append(f"- {LABEL[model]} {DSN[ds]} H={h}: MSE {r[0]:.3f} vs PRISM {p[0]:.3f}; MAE {r[1]:.3f} vs PRISM {p[1]:.3f}")
md.append("\n## Full table (MSE / MAE; lit-96 in brackets)\n")
hdr = "| Dataset | H | PRISM | " + " | ".join(LABEL[m] for m in MODELS) + " |"
md.append(hdr); md.append("|" + "---|" * (3 + len(MODELS)))
for ds in DS:
    for i, h in enumerate(HS):
        p = PRISM[ds][i]
        cells = [f"{p[0]:.3f} / {p[1]:.3f}"]
        for model in MODELS:
            r = RES.get((model, ds, h))
            lit_key = "PaiFilter" if model == "TexFilter" else model
            lit = LIT[lit_key][ds][i]
            c = "–" if r is None else f"{r[0]:.3f} / {r[1]:.3f}"
            if lit and model != "TexFilter": c += f" [{lit[0]:.3f} / {lit[1]:.3f}]"
            cells.append(c)
        md.append(f"| {DSN[ds]} | {h} | " + " | ".join(cells) + " |")
open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n")
json.dump(dict(summary=summary, results={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in RES.items()},
               prism=PRISM, lit=LIT), open(f"{OUT}/ctx336_table.json", "w"), indent=1)
print("\n".join(md[:12]))
print("missing:", [(m, d, h) for m in MODELS for d in DS for h in HS if (m, d, h) not in RES
                   and not (m == "FreqCycle" and d == "exchange") and not (m == "TexFilter" and d not in ("weather", "electricity", "traffic"))])
