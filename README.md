# CT-Encoders: fast self-supervised pretraining of 3D CT encoders

Self-supervised pretraining of encoders for computed tomography, with an iBOT teacher-student objective (DINO
self-distillation plus masked image modelling) and a 3D Swin Transformer backbone, the SwinUNETR encoder, so
the weights drop into SwinUNETR or nnU-Net downstream. This repository continues an existing codebase and makes
its training about **5× faster**, with the same model and the same results.

## Built on the work of

- **[mpizarrot/CT-Encoders](https://github.com/mpizarrot/CT-Encoders)**, by Marcelo Pizarro: the original
  project. It adapted [iBOT](https://github.com/bytedance/ibot) and [MAE](https://github.com/facebookresearch/mae)
  to CT volumes, and added the 3D Swin backbone (`ibot/models/swin3D.py`, based on
  [MONAI](https://github.com/Project-MONAI/MONAI)'s SwinUNETR), the CT datasets and a ViT-LoRA variant.
- **[sebagonzmezz/CT-Encoders](https://github.com/sebagonzmezz/CT-Encoders)**, by Sebastián González: fork of the
  above. It added 3D block and random masking for the MIM loss, the CT augmentation pipeline
  (`transforms.py`, `SpatialTransformRatioCrop.py`, built on nnU-Net's batchgeneratorsv2), the Swin3D
  patch-merging fix, and the probabilistic CLS loss (`--compute_cls_loss_prob`).

This repository starts from their last commit (`aa5393e`), and its git history keeps their commits. Everything
after that commit is described below and in [`david-ct/`](david-ct/).

## The speed-up, in brief

Training was data-bound: CPU augmentation took 0.8–3.5 s per sample per core, and the GPU waited. The window
attention also materialized a 343×343 score matrix per head and window, so a step used 50 GB. Measured on
1,019 real CT volumes, batch 6, one RTX PRO 6000 Blackwell (96 GB):

| Step | s/iteration | Gain | GPU util. | Peak memory |
|---|---|---|---|---|
| Original code, `.nii.gz` files, CPU augmentation | 1.214 | 1× | 58% | 50.5 GB |
| Fused window attention + leaner training loop (bf16) | 0.443 | 2.7× | 70% (now CPU-bound) | 13.9 GB |
| + augmentation on the GPU | 0.361 | 3.4× | 98% | 13.9 GB |
| + `torch.compile` | 0.240 | 5.1× | 99% | 13.2 GB |
| + int16 1.25 mm cache, HU window, crops inside the body (`train_fast.sh`) | **0.231** | **5.3×** | 100% | 13.2 GB |

- 200 epochs of the 13,771-volume CT-Encoders-Dataset: **~29 h instead of ~6.5 days**.
- The model is numerically equivalent to the original (≤1e-5 in fp32; fp16 gradients identical). Checkpoints load
  strictly into the original code. The GPU crops are the CPU crops (correlation 1.0000).
- Speed changes are on by default; options that change what the model sees are opt-in, and `train_fast.sh` turns
  them on.

**The full write-up is [david-ct/CODE.md](david-ct/CODE.md):** every code change file by file with its measured
effect (§3), every benchmark and why the original was slow (§4), and how equivalence was validated (§5).

## Other changes and findings

- **Bug fix:** since commit `eb28c32`, local crops were computed but never reached the CLS loss, so every run with
  local crops trained as if it had none. Fixed in `iBOTLoss` (CODE.md §3.8).
- **Preprocessed cache:** `tools/build_ct_cache.py` rebuilds CT-Encoders-Dataset as int16 volumes at 1.25 mm
  in-plane, cropped to the body. It recovers the spacing and slice axis that the dataset's conversion lost
  (13,771 of 13,771 volumes, 514 → 248 GB).
- **Use the cache with the HU-window recipe.** On the cache, the original recipe (per-volume min-max, crops
  anywhere) collapsed in 2 of 3 short runs, and the HU-window recipe in 0 of 5. Batch 32 collapsed in 3 of 3 runs
  and was not faster, so training stays at batch 6. A low CLS loss alone is not success; `log.txt` now carries
  teacher collapse monitors (CODE.md §3.7 and §7).

## Quick start

```bash
python3 -m venv ~/venvs/ctenc && source ~/venvs/ctenc/bin/activate
pip install torch            # a CUDA build that supports your GPU
pip install -r requirements.txt

# build the cache once (options and inputs: python tools/build_ct_cache.py --help)
python tools/build_ct_cache.py --files names.txt --src /path/to/CT-Encoders-Dataset --out /path/to/cache/ct-encoders

# train: fast pipeline + HU-window recipe, batch 6, 2 global crops
DATA=/path/to/cache/ct-encoders.txt OUT=runs/my_run bash train_fast.sh
```

- `train.sh` is the original recipe on the original `.nii.gz` files, kept as the reference.
- Watch the collapse monitors in `runs/my_run/log.txt` (CODE.md §7); `python tools/diagnose_collapse.py runs/my_run`
  checks a checkpoint's features. Short comparison runs: `python tools/run_queue.py N0 A0`, then
  `python tools/analyze_runs.py`.
- `python tools/check_equivalence.py {model,local-crop-loss,crop-geometry,aug-stats}` re-runs the equivalence
  checks against the original code.
- Upstream entry points still work: `python ibot/main_ibot.py --help` (iBOT, ViT or Swin3D) and
  `python mae/main_pretrain.py --help` (MAE).

## Layout

```
ibot/            iBOT training: main_ibot.py, loader.py, transforms, gpu_transforms.py, models/
mae/             MAE pretraining (from the original project, unchanged)
tools/           cache building, run queue and analysis, benchmarks, equivalence checks
train.sh         original recipe (reference)
train_fast.sh    the recipe to use
david-ct/        CODE.md (the speed-up write-up) and metadata/ (geometry inputs for the cache builder)
runs/            local outputs (not tracked)
```

Maintained by David Miranda.
