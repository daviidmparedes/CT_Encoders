#!/usr/bin/env bash
# Original lab recipe, kept as the reference (use train_fast.sh). Hyper-parameters unchanged from the lab's
# script: per-volume min-max, crops anywhere, CPU augmentation, fp16, 2 global + 0 local crops, batch 6.
# It was trained on the native-resolution .nii.gz files (the CT-Encoders-Dataset on the NAS). Training from the
# NAS is slow and loader-bound; on the 1.25 mm cache this recipe collapsed in 2 of 3 runs
# (david-ct/CODE.md §3.7), so the cache goes with train_fast.sh only.
DATA=${DATA:-/home/shared_data/CT-Encoders-Dataset}
OUT=${OUT:-runs/output_dir_raw_cls_1_lr_00256}

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
  --output_dir "$OUT"
