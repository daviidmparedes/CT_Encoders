"""Run analysis: CLS escape and collapse monitors per run, against samples seen.

Usage: python tools/analyze_runs.py [RUNS_DIR]   (default runs/cross-test)
Reads each <arm>_s<seed>/stdout.log (windowed medians every --log_freq iterations) and log.txt (epoch means).
Writes RUNS_DIR/summary.md and RUNS_DIR/curves.png; the lab's runs are drawn as per-epoch references.

Escape = first logged point from which the 20-iteration median CLS stays below ESCAPE (8.5; ln 8192 = 9.01)
for the next 5 logged points. Measured in samples seen, so batch 6 and batch 32 are comparable.
Note: 2+4 runs have 10 CLS terms (local ones included), 2+0 runs have 2: their CLS values are not comparable.
State per epoch (last one in the table):
- "uniform collapse": the teacher target is uniform (t_ent within 0.05 of ln K). t_protos is then meaningless:
  the argmax of a flat vector is float noise.
- "mode collapse": the epoch-mean target lives on fewer than ~64 effective prototypes (t_mean_ent < ln 64), i.e.
  every volume gets (nearly) the same sharp target. A low CLS loss then means matching that one target.
- "few modes": <= 16 argmax prototypes but a broad epoch-mean target; else "diverse".
diagnose_collapse.py checks the features themselves.
"""
import glob, json, math, os, re, sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "runs", "cross-test")
LAB_LOGS = {  # lab runs on native-resolution files, original code, batch 6, 2+0
    "lab original run, lr 0.0005": "/home/shared_data/CT-Encoders-Dataset-Output/log.txt",  # the nnU-Net encoder
    "lab train.sh run, lr 0.0256": os.path.join(REPO, "runs", "lab-ref", "swin_raw_cls_1_lr_00256", "log.txt"),  # = S's script
}
ESCAPE, HOLD = 8.5, 5
LN_K = math.log(8192)
LINE = re.compile(r"Epoch: \[(\d+)/\d+\]\s+\[\s*(\d+)/(\d+)\]")
METRIC = re.compile(r"(\w+): ([-\d.e]+) \(([-\d.e]+)\)")


def parse(run):
    cmd = json.load(open(os.path.join(run, "cmd.json")))["cmd"]
    batch = int(cmd[len(cmd) - 1 - cmd[::-1].index("--batch_size_per_gpu") + 1])
    pts = []
    for line in open(os.path.join(run, "stdout.log")):
        m = LINE.search(line)
        if not m:
            continue
        ep, it, n = map(int, m.groups())
        vals = {k: float(med) for k, med, _ in METRIC.findall(line)}
        pts.append(dict(samples=(ep * n + it + 1) * batch, **vals))
    epochs = [json.loads(l) for l in open(os.path.join(run, "log.txt"))] if os.path.exists(
        os.path.join(run, "log.txt")) else []
    finished = "Reached --stop_epoch" in open(os.path.join(run, "stdout.log")).read()
    return batch, pts, epochs, finished


def escape(pts):
    c = [p.get("cls", np.nan) for p in pts]
    for i in range(len(c) - HOLD):
        if all(x < ESCAPE for x in c[i:i + HOLD + 1]):
            return pts[i]["samples"]
    return None


runs = sorted(d for d in glob.glob(os.path.join(RUNS, "*_s*")) if os.path.exists(os.path.join(d, "stdout.log")))
data = {os.path.basename(r): parse(r) for r in runs}
labs = {k: [json.loads(l) for l in open(v)][:10] for k, v in LAB_LOGS.items() if os.path.exists(v)}



def state(eps):
    if not eps or "train_t_ent" not in eps[-1]:
        return ""
    e = eps[-1]
    if e["train_t_ent"] > LN_K - 0.05:
        return "uniform collapse"
    if e["train_t_mean_ent"] < math.log(64):
        return "mode collapse"
    if e.get("train_t_protos", 1 << 30) <= 16:
        return "few modes"
    return "diverse"


rows = ["| Run | Batch | Epochs | Escape (samples) | State (last epoch) | Epoch cls | Epoch t_ent | Epoch t_protos | "
        "Epoch t_mean_ent |", "|---|---|---|---|---|---|---|---|---|"]
for name, (batch, pts, eps, finished) in data.items():
    esc = escape(pts)
    fmt = lambda k, f: " → ".join(f.format(e[f"train_{k}"]) for e in eps) if eps and f"train_{k}" in eps[0] else ""
    rows.append(f"| {name} | {batch} | {len(eps)} | {esc if esc else ('never' if finished else 'not yet')} | "
                f"{state(eps)} | {fmt('cls', '{:.2f}')} | "
                f"{fmt('t_ent', '{:.2f}')} | {fmt('t_protos', '{:d}')} | {fmt('t_mean_ent', '{:.2f}')} |")
for name, lab in labs.items():
    lab_cls = " → ".join("{:.2f}".format(e["train_cls"]) for e in lab)
    rows.append(f"| {name} (reference) | 6 | {len(lab)} | | | {lab_cls} | | | |")
summary = "\n".join(rows)
with open(os.path.join(RUNS, "summary.md"), "w") as f:
    f.write(f"# Cross-test summary\n\nEscape threshold {ESCAPE} held for {HOLD} logged points.\n\n{summary}\n")
print(summary)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(S="k", F="tab:blue", N="tab:green", A="tab:orange", B="tab:red")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, (batch, pts, eps, _) in data.items():
        c = colors.get(name[0], "gray")
        x = [p["samples"] for p in pts]
        axes[0].plot(x, [p.get("cls", np.nan) for p in pts], color=c, lw=0.8, label=name)
        axes[1].plot(x, [p.get("t_ent", np.nan) for p in pts], color=c, lw=0.8, label=name)
        if eps:
            n_ep = len(eps)
            spe = x[-1] / n_ep if x else 1
            axes[2].plot([(i + 1) * spe for i in range(n_ep)], [e.get("train_t_protos", np.nan) for e in eps],
                         "o-", color=c, ms=3, label=name)
    for (name, lab), m in zip(labs.items(), "sD"):
        spe = 13771
        axes[0].plot([(i + 0.5) * spe for i in range(len(lab))], [e["train_cls"] for e in lab], m + "--", color="gray",
                     ms=4, label=f"{name} (epoch means)")
    axes[0].axhline(np.log(8192), color="gray", ls=":", lw=0.8)
    for ax, t in zip(axes, ("CLS loss (20-iter median)", "teacher target entropy t_ent", "prototypes used per epoch")):
        ax.set_title(t)
        ax.set_xlabel("samples seen")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(RUNS, "curves.png"), dpi=90)
except Exception as e:
    print("plot failed:", repr(e))
