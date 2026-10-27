"""Run queue for short training comparisons (e.g. the 10-epoch cross test): arms one after another on the GPU.

Arms are the repo's own launch scripts plus overrides, so they cannot drift from what people actually run:
  S  standard  train.sh as is (CPU aug, fp16, min-max, crops anywhere, 2 global + 0 local, batch 6)
  F  fast      train.sh + --gpu_aug true --amp_dtype bf16 --compile true --num_workers 16   (only the pipeline changes)
  N  new recipe train_fast.sh as is (HU window, crops inside the body; still 2 global + 0 local)
  A  batch 6   train_fast.sh + --local_crops_number 4
  B  batch 32  train_fast.sh + --local_crops_number 4 --batch_size_per_gpu 32 --lr 0.0111 --momentum_teacher 0.979
Adjacent arms S-F-N-A-B differ in exactly one thing. Every arm: same data list, --epochs 200 kept (real schedules), --stop_epoch 10, its own --seed.

Usage: python tools/run_queue.py A0 F0 S0 B0 A1 B1 A2 B2      (arm letter + seed; env RUNS, DATA, PY, STOP_EPOCH)
A finished run (10 lines in log.txt) is skipped; an interrupted one resumes from its checkpoint.pth.
Before each launch it waits while another user's process holds GPU memory (the GPU is shared).
"""
import json, os, re, shlex, subprocess, sys, time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.environ.get("RUNS", os.path.join(REPO, "runs", "cross-test"))
DATA = os.environ.get("DATA", "/home/local_data/ct-v2/v2.0_1p25mm/ct-encoders.txt")
PY = os.environ.get("PY", os.path.expanduser("~/venvs/ctenc/bin/python"))
STOP_EPOCH = int(os.environ.get("STOP_EPOCH", "10"))

OVERRIDES = {
    "S": ("train.sh", []),
    "F": ("train.sh", "--gpu_aug true --amp_dtype bf16 --compile true --num_workers 16".split()),
    "N": ("train_fast.sh", []),
    "A": ("train_fast.sh", "--local_crops_number 4".split()),
    "B": ("train_fast.sh", "--local_crops_number 4 --batch_size_per_gpu 32 --lr 0.0111 --momentum_teacher 0.979".split()),
}


def script_args(script):
    """main_ibot.py arguments of a launch script, without --data_path/--output_dir (set per run)."""
    text = open(os.path.join(REPO, script)).read().replace("\\\n", " ")
    line = next(l for l in text.splitlines() if "main_ibot.py" in l and not l.lstrip().startswith("#"))
    toks = shlex.split(line.split("main_ibot.py", 1)[1])
    out, i = [], 0
    while i < len(toks):
        if toks[i] in ("--data_path", "--output_dir"):
            i += 2
            continue
        out.append(toks[i])
        i += 1
    return out


def other_users_on_gpu():
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    me = os.getuid()
    busy = []
    for row in q.strip().splitlines():
        pid, mem = [x.strip() for x in row.split(",")]
        try:
            uid = os.stat(f"/proc/{pid}").st_uid
        except FileNotFoundError:
            continue
        if uid != me:
            busy.append((pid, mem))
    return busy


def epochs_done(out_dir):
    try:
        return sum(1 for _ in open(os.path.join(out_dir, "log.txt")))
    except FileNotFoundError:
        return 0


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(os.path.join(RUNS, "queue.log"), "a") as f:
        f.write(line + "\n")


def main():
    os.makedirs(RUNS, exist_ok=True)
    for spec in sys.argv[1:]:
        arm, seed = spec[0], int(spec[1:])
        script, extra = OVERRIDES[arm]
        out_dir = os.path.join(RUNS, f"{arm}_s{seed}")
        if epochs_done(out_dir) >= STOP_EPOCH:
            log(f"{spec}: already done, skipped")
            continue
        while other_users_on_gpu():
            log(f"{spec}: waiting, GPU in use by another user {other_users_on_gpu()}")
            time.sleep(600)
        os.makedirs(out_dir, exist_ok=True)
        cmd = [PY, "-u", "-W", "ignore", os.path.join(REPO, "ibot", "main_ibot.py"), *script_args(script), *extra,
               "--data_path", DATA, "--output_dir", out_dir, "--seed", str(seed), "--stop_epoch", str(STOP_EPOCH)]
        with open(os.path.join(out_dir, "cmd.json"), "w") as f:
            json.dump(dict(spec=spec, script=script, cmd=cmd), f, indent=1)
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   MASTER_PORT=str(29700 + sum(map(ord, spec)) % 200))
        log(f"{spec}: start ({script} + {' '.join(extra) or 'no overrides'}), resume from epoch {epochs_done(out_dir)}")
        t0 = time.time()
        with open(os.path.join(out_dir, "stdout.log"), "a") as so:
            rc = subprocess.call(cmd, stdout=so, stderr=subprocess.STDOUT, cwd=REPO, env=env)
        log(f"{spec}: exit {rc} after {(time.time() - t0) / 3600:.2f} h, epochs in log.txt: {epochs_done(out_dir)}")


if __name__ == "__main__":
    main()
