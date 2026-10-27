# Code: the training pipeline, what changed, and how it was made fast

The speed-up write-up: how training works, what changed in the code file by file, the measured effect of each
change, and how equivalence was validated. Builds on <https://github.com/mpizarrot/CT-Encoders> and <https://github.com/sebagonzmezz/CT-Encoders>;
"original code" below means their last commit, `aa5393e`. Last updated 2026-10-27.

## 0. In short

- **Train with `bash train_fast.sh`** (env `DATA`, `OUT` override the defaults). It is `train.sh`'s
  hyper-parameters plus GPU augmentation, bf16, `torch.compile`, and the HU-window recipe, on the 1.25 mm cache.
- **5.3× faster per step** than the original code on local data (1.214 → 0.231 s/iteration at batch 6), GPU
  utilization 58% → 100%, memory 50.5 → 13.2 GB. 200 epochs of CT-Encoders-Dataset: ~29 h instead of ~6.5 days.
- **Numerically equivalent model** (≤1e-5 in fp32; checkpoints load strictly into the original code). Speed-only
  changes are on by default; options that change what the model sees are off by default and switched on in
  `train_fast.sh`.
- **One upstream bug fixed:** since commit `eb28c32` local crops never reached the CLS loss (§3.8).
- `train.sh` is kept as the original lab recipe, for reference.

## 1. The codebase

| Path | What |
|---|---|
| `ibot/main_ibot.py` | iBOT training: student/teacher, losses, schedules, the training loop (entry point) |
| `ibot/loader.py` | datasets: `.nii.gz` or int16 `.npy`, a folder or a `.txt` list; MIM masks (`make_mask`); `CTVolumeMask3D` + `collate_volumes` for GPU augmentation |
| `ibot/transforms.py`, `SpatialTransformRatioCrop.py` | CPU multi-crop augmentation (`DataAugmentationCTiBOT2`, batchgeneratorsv2-based) |
| `ibot/gpu_transforms.py` | **new:** GPU port of `DataAugmentationCTiBOT2` (`GPUAugmentationCTiBOT`) |
| `ibot/models/swin3D.py` | MONAI 3D Swin (the SwinUNETR encoder) adapted for MIM; `head.py` = `iBOTHead` |
| `ibot/extract_backbone_weights.py` | strip a checkpoint to the backbone (student or teacher) |
| `ibot/docexplore_dataloader.py`, ViT/2D-Swin/LoRA models | upstream leftovers, not used for CT |
| `mae/` | upstream MAE port (encoder-decoder ViT, 75% masking, pixel reconstruction); not used here |
| `train.sh` / `train_fast.sh` | original recipe / the one to use |
| `tools/` | cache building, run management and checks (below) |

**Tools** (all `python tools/<name> --help`-style docstrings; run from the repo root with `~/venvs/ctenc`):

| Tool | Purpose |
|---|---|
| `build_ct_cache.py` | build the int16 cache from CT-Encoders-Dataset (spacing recovery, slice axis, clip, downsample-only, body crop) |
| `fill_slice_spacing.py` | fill missing slice spacing from ImagePositionPatient; per-series slice/row/column counts |
| `finalize_ct_cache.py` | after a build: training list, `BUILD.md`, QA figure, failure/determinism gate |
| `run_queue.py` | run training arms one after another (resumable; waits while another user holds the GPU) |
| `analyze_runs.py` | per-run CLS escape, collapse state, `summary.md` and `curves.png` for a runs directory |
| `diagnose_collapse.py` | CPU feature check of a checkpoint: are the backbone features constant across volumes? |
| `bench_train.py` | end-to-end throughput of a real training job (s/iter, samples/s, GPU util, data wait, memory) |
| `check_equivalence.py` | regression checks against the original code: `model`, `local-crop-loss`, `crop-geometry`, `aug-stats` |

## 2. How training works

**iBOT** = DINO self-distillation + masked image modelling. A student network sees augmented, partly masked
views of a CT volume; a teacher (EMA of the student, no gradients) sees unmasked views.

| | Count | Covers | Resized to | Seen by |
|---|---|---|---|---|
| Global crops | 2 | 40–100% of the volume | 224³ | teacher (plain) and student (MIM-masked) |
| Local crops | N (`--local_crops_number`) | 5–40% of the volume | 96³ | student only |

- **CLS loss** (scan level): the student's output for every view must match the teacher's output for the other
  global view; local crops add "local-to-global" terms (DINO's multi-crop). Number of terms: 2 × (1 + N), so CLS
  values of 2+0 and 2+4 runs are **not comparable**.
- **Patch/MIM loss:** the student predicts the teacher's patch tokens at masked positions, global crops only,
  from `--pred_start_epoch` (50 in both launch scripts).
- **Head:** `iBOTHead` MLP 384 → 2048 → 2048 → 256, L2-normalized, weight-normed last layer to 8,192 prototypes,
  shared between CLS and patch. `--freeze_last_layer 3` keeps the last layer frozen for epochs 0–2; its release
  at epoch 3 was the turning point of every run we made (§7).
- **Backbone:** Swin3D, embed 48 → 384 features, patch 4, window 7, ~5.7 M parameters but huge activations
  (175,616 tokens per 224³ crop at stage 1). `--patch_size`, `--window_size` and `--drop_path` are **ignored** for
  `swin3D` (hard-coded 4, 7, 0); the scripts pass matching values, so nothing is wrong, but changing them does
  nothing.
- **`--compute_cls_loss_prob`:** probabilistically skips the CLS loss on some iterations (vanilla iBOT = 1). The
  draw uses `np.random`, so ranks agree only because they share a seed (fragile with multi-GPU).

**Schedules and the batch-size catch:**
- `--lr` is a reference **per 256 samples**, multiplied by batch/256: `train.sh`'s 0.0256 at batch 6 = **6.0e-4**
  peak. (The lab's lr 0.0005 run peaked at 1.2e-5, 51× lower.)
- Defined **per step** (change with batch size): teacher EMA `--momentum_teacher 0.996` (~250 steps), the center
  momentum (0.9, hard-coded in `iBOTLoss`), weight decay. At 13,771 volumes batch 6 makes 2,295 steps per epoch,
  batch 32 makes 430.
- Defined **in epochs** (same in samples): LR warmup (`--warmup_epochs`, default 10, so the LR peaks at epoch 10),
  teacher temperature 0.04 → 0.07 over 30 epochs, `freeze_last_layer` 3, `pred_start_epoch` 50.
- Scaling rules if the batch changes: √ for AdamW LR (Malladi et al., 2022), EMA rule `0.996^(B/6)` (Busbridge et
  al., 2023). In practice batch 32 collapsed even with both (3 of 3 runs, 10 epochs each): stay at batch 6, or move the schedules
  to steps first.

## 3. What changed, file by file

Measured effects refer to the benchmark table in §4 (1,019 real volumes, batch 6).
**Defaults keep the original recipe:** speed changes that are numerically equivalent are on by default; options
that change what the model sees are off by default and switched on in `train_fast.sh`. How each change was
checked is in §5.

### 3.1 `ibot/models/swin3D.py`: window attention and masks

- Window attention through `F.scaled_dot_product_attention`, with the windows folded into the head dimension so the
  shifted-window bias broadcasts instead of being copied per window (`--fused_attn`, default on). **bf16/fp32
  only:** under fp16 the kernel's bias gradient underflows, so fp16 keeps the explicit attention.
- Shift masks cached per input shape; no `.clone()` of the relative-position index in every block.
- The 4 multi-scale LayerNorms (on 56³ tensors) that were computed and then discarded with `return_all_tokens=True`
  are skipped.
- The mask embedding uses `torch.where` instead of boolean-index assignment (compile-friendly).
- **Effect:** the biggest single win. At 224³ the old code materialized a 343×343 score matrix per head, window and
  crop at stage 1 (175,616 tokens per crop). Peak memory 50.5 → 13.9 GB. Together with §3.2's loop changes the
  step went 0.717 → 0.443 s/it at equal loader settings (E1 → F2), and F2 was already CPU-bound at 70% GPU.
  Outputs match the original to ≤1e-5.

### 3.2 `ibot/main_ibot.py`: the training loop

- **Bug fix:** local crops reach the CLS loss again (§3.8).
- **GPU augmentation wiring (`--gpu_aug`, off):** workers return raw volumes (`CTVolumeMask3D`), and the crops are
  made on the GPU (§3.5). Effect: 0.443 → 0.361 s/it, GPU 70% → 98% (F2 → F3).
- **`torch.compile` of both backbones (`--compile`, off):** in place, static shapes (one graph per crop size), so
  checkpoint keys are unchanged. Effect: 0.361 → 0.240 s/it (F3 → F6, 1.5×); first compile ~2.5 min, ~35 s from
  cache.
- **`--amp_dtype bf16` (default fp16):** no GradScaler, and it allows the fused attention. With compile: 0.240 s/it
  and 13.2 GB in bf16 vs 0.385 s/it and 34.5 GB in fp16 (F6 vs F5). Autocast now honours `--use_fp16 false`.
- **Fewer host syncs:** metrics stay on the GPU and are copied once every `--log_freq` iterations (default 20; the
  console medians are unchanged); `torch.cuda.synchronize()` removed; the NaN check runs at each flush.
- **Fused AdamW, foreach teacher EMA (same formula), TF32** for the fp32 parts. Part of E1 → F2.
- **Loader options:** `--persistent_workers` (on), `--prefetch_factor 4`, a worker init that limits ITK to one
  thread, and `--pin_memory` (default off with `--gpu_aug`, on otherwise). Pinned buffers stalled the batch-32
  loader for minutes (§6); without them the same run took 1.90 h instead of 3.60 h.
- **Recipe flags** `--intensity_norm`, `--hu_window`, `--crop_inside` (off by default, §3.7).
- **Run control and monitors:** `--stop_epoch N` (stop after epoch N's checkpoint, keeping the `--epochs`
  schedules), `--max_iters` (benchmarks; stops without saving), and the teacher collapse monitors `t_ent`,
  `t_protos`, `t_mean_ent` in `log.txt` (§7).

### 3.3 `ibot/utils.py`

- `clip_gradients_foreach`: the same per-parameter clipping with `torch._foreach_*`, without the ~150 `.item()`
  host syncs per step of `clip_gradients`. Part of E1 → F2.
- An exported `MASTER_PORT` is honoured, so two single-GPU runs no longer collide on port 29500.

### 3.4 `ibot/loader.py`

- Volumes can be int16 `.npy` (the cache) or `.nii.gz`; `--data_path` can be a folder or a `.txt` list of paths
  (relative to the list).
- `CTVolumeMask3D` + `collate_volumes`: the dataset for GPU augmentation (loads the volume and makes the MIM masks;
  no CPU crops). The mask code moved into `make_mask`, logic unchanged.
- The epoch lives in shared memory (`multiprocessing.RawValue`): persistent workers keep their own dataset copy,
  so `set_epoch()` would never reach them and MIM would never start (§5).
- **Effect:** loading a median volume takes 0.020 s from int16 `.npy` vs 0.105 s from `.nii.gz` (5×, one CPU
  core). With GPU augmentation the loader delivers 211–308 volumes/s against the ~26 needed, so raw files and the
  cache train at the same speed; the cache matters for the NAS, disk space and the recipe.

### 3.5 `ibot/gpu_transforms.py` (new): `GPUAugmentationCTiBOT`

- A GPU port of `DataAugmentationCTiBOT2`: the same crop sampling, ranges, probabilities and order. Each crop is
  one `affine_grid` + `grid_sample` (crop, rotation, scaling and resize in one interpolation), then intensity
  scaling, noise, blur, gamma/contrast, low-resolution simulation and mirroring as tensor ops.
- Crops one volume at a time: padding the batch to its largest volume cost ~15 GB of temporaries at batch 32
  (augmentation memory for 32 volumes 20.6 → 5.5 GiB).
- **Effect:** F2 → F3 above. Crop geometry identical to the CPU transform (correlation 1.0000); crop statistics
  match.

### 3.6 `ibot/SpatialTransformRatioCrop.py` and `ibot/transforms.py`: the CPU path

- Rotated/scaled crops sample the output grid directly: one interpolation instead of a grid at the intermediate
  crop size (up to ~80 M points) followed by a resize. Same mapping, slightly less blur. **Effect:** 1.3–4× faster
  on CPU (1,818 ms → 380–450 ms per rotated crop of a 300-slice volume).
- `keep_crop_inside` (crop centres keep the crop inside the volume) and `HUWindowTransform`, with
  `intensity_norm` / `hu_window` / `crop_inside` arguments on `DataAugmentationCTiBOT2`. Recipe, off by default.

### 3.7 Recipe options (they change the experiment)

- **`--intensity_norm hu_window`** (`--hu_window -1000 1000` → [0, 1]) instead of per-volume min-max. Min-max
  gives each source its own contrast curve, a dataset-identity shortcut, and on the cache the original recipe
  collapsed in 2 of 3 ten-epoch runs, against 0 of 5 with the HU window. Required before mixing AbdomenAtlas
  (pre-clipped to ±1000).
- **`--crop_inside true`**: zero voxels per crop drop from ~45% to ~12%, and those are real air. In the original
  recipe a median 56% of every global crop is padding outside the scan.

### 3.8 The upstream bug: local crops never reached the CLS loss

Commit `eb28c32` (2025-10-29, probabilistic CLS loss) changed the CLS pairing loop in `iBOTLoss.forward` from
`student_cls_c` to `student_patch_c`. Patch outputs exist only for the 2 global crops, so every local crop was
computed and dropped with zero gradient. 2+0 runs (`train.sh`, the lab's runs) are unaffected; every 2+N run since
then trained like 2+0. The loop runs over all student views again, as in upstream iBOT. It shows in the lab's runs:
the "2+4" `…_b18_g2_l4` run has the same CLS curve as the 2+0 run.

### 3.9 Tools and launch scripts

`tools/` (§1) and `train_fast.sh` are new. `train.sh` keeps the lab's hyper-parameters; only its data path changed.
`main_ibot.py`, `loader.py` and `utils.py` use CRLF line endings; keep them so diffs show real changes only.

## 4. Why it was slow, and what made it fast

**Audit (2026-10-05, CPU-only while the GPU was busy):** training was data-bound. The CPU augmentation took
0.8–3.5 s per sample per core (two 224³ crops), ~45% of worker time was kernel time from ~1 M page faults per
sample, the affine crop built its grid at the intermediate crop size and resized twice, and the original pipeline
topped out at 2.6 samples/s with 10 workers even from RAM (≥12 days for 200 epochs). Reading `.nii.gz` from the
NAS was far worse.

**All measurements** (`tools/bench_train.py`: real training of `ibot/main_ibot.py` on 1,019 real CT volumes,
batch 6, `train.sh` hyper-parameters with MIM on, iterations 40→140 after warm-up and compile; one RTX PRO 6000
Blackwell Max-Q, 96 GB):

| Run | What | s/it | samples/s | GPU | Peak mem | vs E0 |
|---|---|---|---|---|---|---|
| E0 | **original code**, local `.nii.gz`, CPU augmentation, 10 workers, fp16 | 1.214 | 4.94 | 58% | 50.5 GB | 1× |
| E1 | original code, 24 workers + `THP_MEM_ALLOC_ENABLE=1` (don't use THP, §6) | 0.717 | | GPU-bound | | 1.7× |
| F2 | new model and training-loop code, CPU augmentation, bf16, 24 workers | 0.443 | 13.54 | 70% | 13.9 GB | 2.7× |
| F3 | + augmentation on the GPU | 0.361 | 16.61 | 98% | 13.9 GB | 3.4× |
| F4 | GPU augmentation in fp16 (explicit attention) | 0.733 | 8.18 | 100% | 51.0 GB | |
| F5 | fp16 + compile | 0.385 | 15.59 | 99% | 34.5 GB | |
| F6 | bf16 + compile | 0.240 | 24.97 | 99% | 13.2 GB | 5.1× |
| **F7** | + 1.25 mm int16 cache, HU window, crops inside the body (= `train_fast.sh`) | **0.231** | **25.98** | 100% | 13.2 GB | **5.3×** |
| E10a | original code, 2 global + 4 local crops | 1.667 | 3.60 | 47% | 55.6 GB | |
| F10b | optimized, 2 global + 4 local crops | 0.267 | 22.46 | 100% | 14.2 GB | 6.2× vs E10a |

1. **The model code was the biggest single win**, mostly the window attention: the old code materialized a 343×343
   score matrix per head, window and crop, plus shifted-mask copies. The fused kernel halves the step and cuts
   memory 50 → 14 GB.
2. **Then the CPU augmentation became the bottleneck** (F2); moving it to the GPU saturates the GPU (F3). Workers
   only load volumes.
3. **`torch.compile` adds 1.5×**: the model is bandwidth-bound (LayerNorms, permutes, window partition/roll).
   First compile ~2.5 min, ~35 s from cache.
4. **Use bf16.** fp16 must use the explicit attention: 0.733 s/it, or 0.385 with compile, at 34–51 GB.

Other measurements:
- The int16 cache vs raw `.nii.gz`: the same speed once augmentation is on the GPU (the loader delivers 211–308
  volumes/s vs ~26 needed). 1.25 vs 1.5 mm cache: 0.231 vs 0.230 s/it. 16 vs 32 workers: the same.
- Batch 12/16/24/32 (2+0): 24.3 / 20.4 / 20.0 / 18.8 samples/s, so no gain (the GPU is full at batch 6). With 2+4
  crops, batch 32 runs at 1.594 s/it and 71.4 GB, 11% slower per sample than batch 6 (0.270 s/it, 14.5 GB).
- E10a and F10b ran while local crops never reached the loss (§3.8), so without their backward. After the fix, 2+4
  at batch 6 is 0.270 s/it and 14.5 GB: the speed figures stand.
- Full dataset (13,771 volumes, 10 epochs): original code path 0.707 s/it (4.45 h, not loader-bound from a local
  int16 cache, 52 GB); fast pipeline 0.240 s/it (1.55 h, 13.6 GB); with 2+4 crops 0.280 s/it (1.80 h). 200 epochs:
  ~29 h instead of ~6.5 days (~34 h with 2+4 crops).

**Not tried yet:** `compile(mode="max-autotune")`, compiling the iBOT head + loss (the 8,192-way softmax over every
patch token is ~8% of the step), CUDA graphs. Expect tens of percent, not multiples.

## 5. Validation

Rerun with `tools/check_equivalence.py` (it extracts the original code with `git archive aa5393e ibot`).

| Check | Result |
|---|---|
| `model`: optimized vs original Swin3D, fp32, 224³ with MIM mask | max output diff 3.5e-6 (scale 4.7), max relative grad diff 9.6e-6; 96³: 2.4e-6 / 3.4e-6; multi-scale outputs ≤2.4e-6 |
| `model`: relative-position-bias gradients under autocast | fp16 **identical**; bf16 2.3e-2 (bf16 precision) |
| `crop-geometry`: GPU crop vs `SpatialTransformRatioCrop` with the same parameters | crop+resize correlation 1.0000 (rel. error 0.0002); rotation/scaling identical to the new CPU path, 0.9996 vs the original's two-interpolation path |
| `aug-stats`: 80 crops, CPU vs GPU augmentation | mean 0.112 / 0.115, std 0.161 / 0.166, padding 0.45 / 0.40 |
| `local-crop-loss` | 2+0 bit-identical; 2+4: original gives local crops exactly zero gradient, fixed uses them, patch loss unchanged |
| Checkpoints | state-dict keys unchanged under compile; load strictly into the original model |
| Per-volume GPU cropping (2026-10-13) | same crops as the padded batch to float rounding (4e-5); augmentation memory for 32 volumes 20.6 → 5.5 GiB |

**Bugs caught during the work (both fixed):**
1. Persistent workers keep their own dataset copy, so `set_epoch()` never reached them: with
   `--pred_start_epoch 50`, MIM would never have started. The epoch now lives in a `multiprocessing.RawValue`
   (checked with fork, forkserver and spawn).
2. Fused attention under fp16 returned ~zero relative-position-bias gradients (|g| ~1e-7 vs ~1e-4). It did not
   show in fp32 tests; it showed as a CLS loss stuck at ln 8192. **Lesson: check gradients under the real
   autocast dtype, plus a short multi-seed run, for any kernel change.**

## 6. Pitfalls on this machine

- **Don't use `THP_MEM_ALLOC_ENABLE=1`.** It gave +54–66% loader throughput on fresh memory, but with THP defrag =
  madvise and long uptime it later caused machine-wide memory-compaction stalls (every worker at 100% system time,
  runs stalled 15+ min, other jobs slowed too).
- **Pinned memory + GPU augmentation at large batch:** workers return raw volumes of many sizes; at batch 32
  (16 workers × prefetch 4 = up to 2,048 volumes in flight) the pinned allocator grew shared memory 42 → 82 GB in
  2 min and the loader stalled 63–588 s at a time. Pinning saves <1% of a step (9 vs 19 ms per copy). Hence
  `--pin_memory` defaults off with `--gpu_aug`.
- **Batch 32 needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (set in `train_fast.sh`): the fused
  attention backward asks for single 9–24 GB blocks and a fragmented cache OOMs at step 2. Saved activations are
  66 GB at batch 32 with 2+4 crops (peak 70.5 GB); batch 6 uses 14 GB.
- **`torch.compile` needs the Python headers** (`python3.12-dev`, installed).
- **Running two jobs:** export distinct `MASTER_PORT`s.
- **The GPU is shared:** `tools/run_queue.py` waits while another user's process holds GPU memory.

## 7. Monitoring a run

`log.txt` has, per epoch, `train_cls`, `train_patch` and three teacher monitors:

| Monitor | Meaning |
|---|---|
| `t_ent` | mean entropy of the teacher's CLS target; ln 8192 = 9.01 = uniform, ~0 = one-hot |
| `t_protos` | prototypes used by the teacher's argmax this epoch |
| `t_mean_ent` | entropy of the epoch-mean target (how spread the targets are across volumes) |

- **Healthy:** low `t_ent`, many prototypes, high `t_mean_ent`.
- **Uniform collapse:** `t_ent` = ln K and the CLS loss exactly 9.011. `t_protos` is then meaningless (argmax of
  a flat vector is float noise). The backbone gives one constant feature; the run cannot recover.
- **Mode collapse:** `t_mean_ent` < ln 64: every volume gets nearly the same sharp target, so a **low CLS loss is
  not a success signal by itself**. It happened temporarily in epochs 1–2 of many healthy runs.
- **The epoch-3 release of the head's last layer** (`--freeze_last_layer 3`) is the turning point: healthy runs
  leave their mode collapse there, while runs whose features are not yet diverse (batch 32) collapse there for good.
- `python tools/analyze_runs.py RUNS_DIR` classifies the last epoch (uniform collapse / mode collapse / few modes /
  diverse) and plots curves against samples seen. `GLOBAL=1 python tools/diagnose_collapse.py RUN_DIR` (CPU,
  seconds) checks the backbone itself: mean pairwise cosine of the CLS feature over 8 volumes (1.0 = collapsed)
  on 224³ crops, the teacher's actual view (96³ crops can mislead).
- Logged `data:` vs `time:` per iteration shows whether a run is loader-bound.

## 8. Environment

`~/venvs/ctenc` (Python 3.12): torch 2.14.1+cu130, monai 1.4.0, SimpleITK 2.4.1, batchgeneratorsv2 0.3.0,
timm 1.0.12, einops, tensorboardX 2.6.2, numpy 1.26.4, scipy 1.17.1, nibabel 5.4.2, pyarrow 25.0.1,
matplotlib 3.11.2. GPU: RTX PRO 6000 Blackwell Max-Q, 96 GB (sm_120, needs a CUDA ≥ 12.8 build). wheeljack:
EPYC 9355 (32 cores / 64 threads), 188 GB RAM.
