"""Finish a cache build: training list, BUILD.md provenance, QA figure, determinism check.

Usage: python tools/finalize_ct_cache.py OUT_DIR BUILD_DIR [EARLIER_CACHE_DIR]
  e.g. python tools/finalize_ct_cache.py /home/local_data/ct-v2/v2.0_1p25mm/ct-encoders runs/cross-test/build
Writes <out>/../ct-encoders.txt (successful volumes, relative paths), <out>/BUILD.md, <out>/qa/mid_slices.png.
"""
import collections, glob, hashlib, json, os, subprocess, sys, time

import numpy as np

out, build = sys.argv[1], sys.argv[2]
repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# optional: an earlier build of some of the same volumes (same builder and args), which must be bit-identical
SAMPLE_CACHE = sys.argv[3] if len(sys.argv) > 3 else None

expected = [l.strip() for l in open(os.path.join(build, "all.txt")) if l.strip()]
rows = {}
for line in open(os.path.join(out, "manifest.jsonl")):
    r = json.loads(line)
    rows[r["name"]] = r  # a retried file appends a new row: the last one wins
ok = {n: r for n, r in rows.items() if "error" not in r and os.path.exists(os.path.join(out, r["path"]))}
failed = {n: r.get("error", "output missing") for n, r in rows.items() if n not in ok}
missing = [n for n in expected if n not in rows]
leftovers = glob.glob(os.path.join(out, "*.tmp.npy"))

# training list: paths relative to the list's directory (loader.list_volumes)
name = os.path.basename(out.rstrip("/"))
list_path = os.path.join(os.path.dirname(out.rstrip("/")), f"{name}.txt")
with open(list_path, "w") as f:
    for n in sorted(ok):
        f.write(f"{name}/{ok[n]['path']}\n")

# determinism: the sample built in the speed tests vs this build
same = diff = 0
for p in sorted(glob.glob(os.path.join(SAMPLE_CACHE, "*.npy"))) if SAMPLE_CACHE else []:
    q = os.path.join(out, os.path.basename(p))
    if os.path.exists(q):
        if np.array_equal(np.load(p), np.load(q)):
            same += 1
        else:
            diff += 1

by_src = collections.Counter(r["source"] for r in ok.values())
fail_src = collections.Counter(n.split("_")[0] for n in failed)
gb = sum(r["mb"] for r in ok.values()) / 1e3
zsp = np.array([r["spacing_zyx"][0] for r in ok.values()])
shapes = np.array([r["shape_zyx"] for r in ok.values()])
methods = collections.Counter(r.get("spacing_method") for r in ok.values())
git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()
sha = hashlib.sha256(open(os.path.join(repo, "tools", "build_ct_cache.py"), "rb").read()).hexdigest()[:16]
times = open(os.path.join(build, "build_times.txt")).read().strip()

# QA figure: axial / coronal / sagittal mid-slices, 2 volumes per source, HU window [-1000, 1000]
qa = os.path.join(out, "qa")
os.makedirs(qa, exist_ok=True)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    picks = []
    for src in sorted(by_src):
        names = sorted(n for n, r in ok.items() if r["source"] == src)
        picks += [names[i] for i in rng.choice(len(names), size=min(2, len(names)), replace=False)]
    fig, axes = plt.subplots(len(picks), 3, figsize=(7.5, 2.5 * len(picks)))
    for row, n in zip(axes, picks):
        v = np.load(os.path.join(out, ok[n]["path"]), mmap_mode="r")
        sp = ok[n]["spacing_zyx"]
        z, y, x = (s // 2 for s in v.shape)
        for ax, img, aspect, title in ((row[0], v[z], sp[1] / sp[2], "axial"),
                                       (row[1], v[:, y, :], sp[0] / sp[2], "coronal"),
                                       (row[2], v[:, :, x], sp[0] / sp[1], "sagittal")):
            ax.imshow(np.asarray(img), cmap="gray", vmin=-1000, vmax=1000, aspect=aspect)
            ax.set_axis_off()
            ax.set_title(f"{title}" if title != "axial" else f"{n[:40]}\n{tuple(v.shape)} {sp[0]:.2f} mm", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(qa, "mid_slices.png"), dpi=80)
    qa_note = f"`qa/mid_slices.png` ({len(picks)} volumes, 2 per source)"
except Exception as e:  # the figure is a convenience, not a gate
    qa_note = f"QA figure failed: {e!r}"

with open(os.path.join(out, "BUILD.md"), "w") as f:
    f.write(f"""# {name} cache (1.25 mm, int16, body-cropped)

Written by tools/finalize_ct_cache.py on
{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}.

- Builder: tools/build_ct_cache.py sha256 {sha}; repo HEAD {git('rev-parse', '--short', 'HEAD')} on branch
  {git('rev-parse', '--abbrev-ref', 'HEAD')} ({'uncommitted changes' if git('status', '--porcelain') else 'clean'})
- Command: tools/build_ct_cache.py (spacing 1.25, body crop, margin 10 mm, native index + filled geometry)
- Times: {times.replace(chr(10), '; ')}
- Volumes: {len(ok)} ok / {len(expected)} expected; failed {len(failed)}; never attempted {len(missing)};
  leftover .tmp.npy {len(leftovers)}
- Size: {gb:.1f} GB. Training list: {list_path} ({len(ok)} lines)
- Determinism vs an earlier build: {f'{SAMPLE_CACHE}: {same} identical, {diff} different' if SAMPLE_CACHE else 'not checked'}
- Spacing source: {dict(methods)}
- Slice spacing after resampling (mm): min {zsp.min():.2f}, median {np.median(zsp):.2f}, max {zsp.max():.2f};
  {(zsp > 1.3).mean() * 100:.1f}% thicker than 1.3 mm (thick slices are never upsampled)
- Shape (z, y, x) median {np.median(shapes, 0).astype(int).tolist()}, max {shapes.max(0).tolist()}
- QA: {qa_note}

| Source | ok | failed |
|---|---|---|
""")
    for src in sorted(set(by_src) | set(fail_src)):
        f.write(f"| {src} | {by_src.get(src, 0)} | {fail_src.get(src, 0)} |\n")
    if failed:
        f.write("\nFailures:\n\n" + "".join(f"- `{n}`: {e}\n" for n, e in sorted(failed.items())))
# machine-readable verdict, to gate anything that should only start on a good build
gate = dict(ok=len(ok), expected=len(expected), failed=len(failed), missing=len(missing), leftovers=len(leftovers),
            sample_identical=same, sample_different=diff, gb=round(gb, 1), list=list_path)
gate["pass"] = (len(failed) <= 0.01 * len(expected) and not missing and not leftovers and diff == 0 and (same > 0 or not SAMPLE_CACHE))
json.dump(gate, open(os.path.join(build, "build_summary.json"), "w"), indent=1)
print(open(os.path.join(out, "BUILD.md")).read())
print("gate:", gate)
