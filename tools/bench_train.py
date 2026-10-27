"""End-to-end training throughput benchmark for ibot/main_ibot.py.

Runs a real training job and measures, over a steady-state window of iterations [--start, --stop):
  wall seconds/iteration (from log-line arrival times), samples/s, GPU utilization (nvidia-smi),
  the logger's mean data-wait per iteration, and peak GPU memory. The job is killed at --stop.

Example:
  python tools/bench_train.py --name E0_baseline --python ~/venvs/ctenc/bin/python \
      -- --data_path /home/local_data/ct-v2/v2.0_1p25mm/ct-encoders.txt --num_workers 10 ...
Appends one JSON line to --out (default runs/bench/results.jsonl). The recorded results are in david-ct/CODE.md §4.
"""
import argparse, json, os, re, signal, subprocess, sys, threading, time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LINE = re.compile(r"Epoch: \[(\d+)/\d+\]\s+\[\s*(\d+)/(\d+)\].*?time: ([\d.]+)\s+data: ([\d.]+)\s+max mem: (\d+)")

TRAIN_SH = ("--epochs 200 --compute_cls_loss_prob 1 --arch swin3D --patch_size 4 --window_size 7 --lr 0.0256 "
            "--min_lr 1e-06 --norm_last_layer False --global_crops_scale 0.4 1.0 --local_crops_scale 0.05 0.4 --local_crops_number 0 "
            "--clip_grad 3.0 --pred_ratio 0.0 0.3 --pred_ratio_var 0.0 0.2 --pred_shape rand --pred_start_epoch 0 "
            "--warmup_teacher_temp_epochs 30 --saveckp_freq 40").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--code", default=os.path.join(REPO, "ibot"), help="directory containing main_ibot.py (e.g. an "
                    "original tree from `git archive aa5393e ibot`, to compare with)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--start", type=int, default=40)
    ap.add_argument("--stop", type=int, default=140)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "bench", "results.jsonl"))
    ap.add_argument("--logdir", default=os.path.join(REPO, "runs", "bench", "logs"))
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="extra main_ibot.py args after --")
    a = ap.parse_args()
    extra = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    os.makedirs(a.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out_dir = os.path.join(a.logdir, a.name)
    cmd = [a.python, "-u", "-W", "ignore", os.path.join(a.code, "main_ibot.py"), *TRAIN_SH,
           "--batch_size_per_gpu", str(a.batch), "--output_dir", out_dir, *extra]
    env = dict(os.environ, **dict(e.split("=", 1) for e in a.env))

    util, stop_util = [], threading.Event()

    def poll_gpu():
        p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                              "-lms", "500"], stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            if stop_util.is_set():
                break
            util.append((time.time(), float(line.strip())))
        p.kill()

    threading.Thread(target=poll_gpu, daemon=True).start()
    log = open(os.path.join(a.logdir, a.name + ".log"), "w")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                            cwd=os.path.dirname(a.code.rstrip("/")) or ".", start_new_session=True)
    t_start = time.time()
    marks, data_waits, peak_mem, err = {}, [], 0, None
    for line in proc.stdout:
        log.write(line)
        log.flush()
        m = LINE.search(line)
        if m:
            ep, k, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
            it, data, mem = ep * n + k, float(m.group(5)), int(m.group(6))  # global iteration
            marks[it] = time.time()
            peak_mem = max(peak_mem, mem)
            if it >= a.start:
                data_waits.append(data)
            if it >= a.stop:
                break
        if "Traceback" in line or "Error" in line and "Warning" not in line:
            err = (err or "") + line
    os.killpg(proc.pid, signal.SIGTERM)
    stop_util.set()
    proc.wait()
    log.close()
    ks = sorted(k for k in marks if a.start <= k <= a.stop)
    res = dict(name=a.name, batch=a.batch, args=" ".join(extra), env=a.env)
    if len(ks) >= 2:
        k0, k1 = ks[0], ks[-1]
        spi = (marks[k1] - marks[k0]) / (k1 - k0)
        u = [v for t, v in util if marks[k0] <= t <= marks[k1]]
        res.update(s_per_iter=round(spi, 4), samples_per_s=round(a.batch / spi, 2),
                   gpu_util=round(sum(u) / max(1, len(u)), 1), data_wait_s=round(sum(data_waits) / len(data_waits), 4),
                   peak_mem_gb=round(peak_mem / 1024, 1), window=[k0, k1], startup_s=round(marks[ks[0]] - t_start, 1))
    else:
        res.update(error=(err or "no iterations logged")[-2000:])
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
