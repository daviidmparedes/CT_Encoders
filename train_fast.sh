#!/usr/bin/env bash
# Fast iBOT/Swin3D pretraining (use this one): train.sh hyper-parameters, plus
#   - GPU augmentation (workers only load volumes)        --gpu_aug true
#   - bf16 autocast (no GradScaler)                        --amp_dtype bf16
#   - torch.compile of the backbones                       --compile true
#   - recipe: fixed HU window + crops kept inside the body --intensity_norm hu_window --crop_inside true
# Data: the int16 1.25 mm body-cropped cache (tools/build_ct_cache.py).
# What each option buys and how it was checked: david-ct/CODE.md.
# - Stay at batch 6: batch 32 is not faster here and collapsed in 3 of 3 runs.
# - 2+0 crops; 2+4 (--local_crops_number 4) is just as stable and costs +16% time. Which one gives better
#   features is still open (downstream probe).
# torch.compile needs the Python headers (python3.12-dev); without them set --compile false.
DATA=${DATA:-/home/local_data/ct-v2/v2.0_1p25mm/ct-encoders.txt}
OUT=${OUT:-runs/output_dir_fast}
# Expandable segments avoid the allocator fragmentation OOM seen at large batches; no cost at batch 6
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

python3 ibot/main_ibot.py \
  --epochs 200 \
  --batch_size_per_gpu 6 \
  --data_path "$DATA" \
  --compute_cls_loss_prob 1 \
  --arch swin3D \
  --patch_size 4 \
  --window_size 7 \
  --lr 0.0256 \
  --min_lr 1e-06 \
  --norm_last_layer False \
  --global_crops_scale 0.4 1.0 \
  --local_crops_scale 0.05 0.4 \
  --local_crops_number 0 \
  --clip_grad 3.0 \
  --pred_ratio 0.0 0.3 \
  --pred_ratio_var 0.0 0.2 \
  --pred_shape rand \
  --pred_start_epoch 50 \
  --warmup_teacher_temp_epochs 30 \
  --saveckp_freq 40 \
  --gpu_aug true \
  --amp_dtype bf16 \
  --compile true \
  --num_workers 16 \
  --intensity_norm hu_window \
  --crop_inside true \
  --output_dir "$OUT"
