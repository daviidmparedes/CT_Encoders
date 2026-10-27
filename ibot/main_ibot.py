# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from pathlib import Path
import torchvision.transforms as T
from torchvision.ops.misc import Permute
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter
from models.head import iBOTHead
from loader import CTFolderMask, CTFolderMask3D, CTVolumeMask3D, collate_volumes, worker_init_fn
from models.swin3D import swin_3D
from models.vit_lora import vit_lora

from transforms import DataAugmentationCTiBOT2
from gpu_transforms import GPUAugmentationCTiBOT

def get_args_parser():
    parser = argparse.ArgumentParser('iBOT', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_base', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
                 'swin_tiny','swin_small', 'swin_base', 'swin_large', 'swin3D', 'vit_lora'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=True, type=utils.bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=[0.0, 0.7], type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=[0.0, 0.05], type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='rand', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")
    parser.add_argument('--compute_cls_loss_prob', default=1.0, type=float, help="""Probability with which CLS loss 
        is computed during training after warm up epochs (Default: 1.0)""")
        
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=0.3, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=3, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.00075, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=2e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.32, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=10, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.32),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=40, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--ckpt_path_student', default="/home/mpizarro/data/ibot_pretrained_student.pth", type=str)
    parser.add_argument('--ckpt_path_teacher', default="/home/mpizarro/data/ibot_pretrained_teacher.pth", type=str)

    # Speed (defaults keep the original training recipe; see david-ct/CODE.md)
    parser.add_argument('--gpu_aug', type=utils.bool_flag, default=False, help="""swin3D only: workers just load
        the volume and the multi-crop augmentation runs on the GPU (gpu_transforms.py) instead of on the CPU.""")
    parser.add_argument('--amp_dtype', default='fp16', choices=['fp16', 'bf16'], help="""Autocast dtype when
        --use_fp16 is true. bf16 needs no GradScaler.""")
    parser.add_argument('--compile', type=utils.bool_flag, default=False, help="torch.compile the backbones.")
    parser.add_argument('--fused_attn', type=utils.bool_flag, default=True, help="""swin3D: use
        F.scaled_dot_product_attention in the window attention (numerically equivalent, less memory).""")
    parser.add_argument('--persistent_workers', type=utils.bool_flag, default=True)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--pin_memory', type=utils.bool_flag, default=None, help="""Pin the loader's host
        buffers. Default: off with --gpu_aug (raw volumes of many sizes: the pinned-memory cache grows by tens of GB
        and stalls the loader for minutes at batch 32, while the copy it speeds up is <1%% of a step), on otherwise.""")
    parser.add_argument('--log_freq', type=int, default=20, help="Print and sync metrics every N iterations.")
    parser.add_argument('--max_iters', type=int, default=0, help="Stop after N iterations (benchmarks/smoke tests).")
    parser.add_argument('--stop_epoch', type=int, default=0, help="""Stop after this many epochs, with the
        checkpoint and log.txt written; the schedules still follow --epochs (short A/B runs of the real recipe).""")
    # Recipe options (change what the model sees; defaults = original behaviour)
    parser.add_argument('--intensity_norm', default='minmax', choices=['minmax', 'hu_window'], help="""Per-volume
        min-max scaling (original) or a fixed HU window mapped to [0, 1].""")
    parser.add_argument('--hu_window', type=float, nargs=2, default=(-1000., 1000.))
    parser.add_argument('--crop_inside', type=utils.bool_flag, default=False, help="""Keep random crops inside
        the volume (original: centres anywhere, so a global crop is ~56%% zero padding).""")
    return parser

def train_ibot(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    if args.pin_memory is None:
        args.pin_memory = not (args.arch == "swin3D" and args.gpu_aug)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ============ preparing data ... ============
    transform = DataAugmentationCTiBOT2(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
        intensity_norm=args.intensity_norm,
        hu_window=args.hu_window,
        crop_inside=args.crop_inside,
    )
    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size
    gpu_aug, collate_fn = None, None
    if args.arch == "swin3D" and args.gpu_aug:
        dataset = CTVolumeMask3D(
            args.data_path,
            crop_sizes=[224] * args.global_crops_number + [96] * args.local_crops_number,
            patch_size=pred_size,
            pred_ratio=args.pred_ratio,
            pred_ratio_var=args.pred_ratio_var,
            pred_aspect_ratio=(0.3, 1/0.3),
            pred_shape=args.pred_shape,
            pred_start_epoch=args.pred_start_epoch)
        gpu_aug = GPUAugmentationCTiBOT(
            args.global_crops_scale,
            args.local_crops_scale,
            args.global_crops_number,
            args.local_crops_number,
            intensity_norm=args.intensity_norm,
            hu_window=args.hu_window,
            crop_inside=args.crop_inside,
            seed=args.seed + utils.get_rank())
        collate_fn = collate_volumes
    elif args.arch == "swin3D":
        dataset = CTFolderMask3D(
            args.data_path, 
            transform=transform,
            patch_size=pred_size,
            pred_ratio=args.pred_ratio,
            pred_ratio_var=args.pred_ratio_var,
            pred_aspect_ratio=(0.3, 1/0.3),
            pred_shape=args.pred_shape,
            pred_start_epoch=args.pred_start_epoch,
            batch_size=224)
    else:
        dataset = CTFolderMask(
            args.data_path, 
            transform=transform,
            patch_size=pred_size,
            pred_ratio=args.pred_ratio,
            pred_ratio_var=args.pred_ratio_var,
            pred_aspect_ratio=(0.3, 1/0.3),
            pred_shape=args.pred_shape,
            pred_start_epoch=args.pred_start_epoch)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    data_loader.gpu_aug = gpu_aug
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if args.arch == "vit_lora":
        student = vit_lora(
            ckpt_path=args.ckpt_path_student,
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = vit_lora(
            ckpt_path=args.ckpt_path_teacher,
            patch_size=args.patch_size,
            return_all_tokens=True,
        )
        embed_dim = student.embed_dim
    elif args.arch == "swin3D":
        student = swin_3D(return_all_tokens=True, 
            masked_im_modeling=args.use_masked_im_modeling,
            fused_attn=args.fused_attn,
        )
        teacher = swin_3D(return_all_tokens=True,
            fused_attn=args.fused_attn,
        )
        embed_dim = student.num_features
    
    elif args.arch in models.__dict__.keys() and 'swin' in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True, 
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True,
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, iBOTHead(
        embed_dim,
        args.out_dim,
        patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head,
        act=args.act_in_head,
        norm_last_layer=args.norm_last_layer,
        shared_head=args.shared_head,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    if args.compile:
        # in-place Module.compile keeps the state_dict keys (checkpoints stay compatible);
        # static shapes: one graph per crop size (global/local) instead of a slow dynamic-shape graph
        student.backbone.compile(dynamic=False)
        teacher.backbone.compile(dynamic=False)
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False) if \
            'swin' in args.arch else nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False) if \
        'swin' in args.arch else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
        compute_cls_loss_prob=args.compute_cls_loss_prob,
    ).cuda()

    if utils.is_main_process(): # Tensorboard configuration
        local_runs = os.path.join(args.output_dir, 'tf_logs')
        writer = SummaryWriter(logdir=local_runs)
        
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups, fused=True)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16 and args.amp_dtype == 'fp16':
        fp16_scaler = torch.amp.GradScaler('cuda')

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(data_loader))
                  
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            args.load_from,
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting iBOT training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of iBOT ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)
        iters_done = train_stats.pop('iters')
        if args.max_iters and len(data_loader) * epoch + iters_done >= args.max_iters:
            print(f"Reached --max_iters {args.max_iters}; stopping without checkpoint.")
            break

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                for k, v in train_stats.items():
                    writer.add_scalar(k, v, epoch)
        if args.stop_epoch and epoch + 1 >= args.stop_epoch:
            print(f"Reached --stop_epoch {args.stop_epoch}; checkpoint saved, stopping.")
            break
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    params_q = [p.data for p in params_q]
    params_k = [p.data for p in params_k]
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16

    # per-iteration metrics stay on the GPU and are copied to the host in one transfer every --log_freq
    # iterations: a .item() per metric (and per parameter in the old gradient clipping) stalls the CPU every step
    pending = {}

    def flush_metrics():
        nonlocal pending
        if not pending:
            return
        values = {k: torch.stack(v).tolist() for k, v in pending.items()}
        pending = {}
        if not all(math.isfinite(x) for x in values['loss']):
            print("Loss is {}, stopping training".format(values['loss']), force=True)
            sys.exit(1)
        for k, vs in values.items():
            for v in vs:
                metric_logger.meters[k].update(v)

    # collapse monitor: which teacher CLS prototypes are used this epoch (summed on the GPU, read once per epoch)
    proto_counts = torch.zeros(args.out_dim, dtype=torch.long, device='cuda')
    proto_prob_sum = torch.zeros(args.out_dim, device='cuda')

    n_iters = len(data_loader)
    it_in_epoch = 0
    for it_in_epoch, (images, masks) in enumerate(metric_logger.log_every(data_loader, args.log_freq, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it_in_epoch  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        if data_loader.gpu_aug is not None:
            # `images` are the raw volumes; crops are made on the GPU
            images = data_loader.gpu_aug([v.cuda(non_blocking=True) for v in images])
        else:
            images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]        
        
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=args.use_fp16):
            # get global views
            teacher_output = teacher(images[:args.global_crops_number])
            student_output = student(images[:args.global_crops_number], mask=masks[:args.global_crops_number])

            # get local views
            student.module.backbone.masked_im_modeling = False
            student_local_cls = student(images[args.global_crops_number:])[0] if len(images) > args.global_crops_number else None
            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch)
            loss = all_loss.pop('loss')

        # collapse monitor on the teacher CLS targets as the loss uses them (centred, sharpened). A low CLS loss alone
        # can mean collapse: t_ent ~ ln(out_dim) = stuck at uniform, ~0 with few prototypes per epoch = one-hot collapse
        with torch.no_grad():
            t_prob = F.softmax((teacher_output[0].float() - ibot_loss.center) / ibot_loss.teacher_temp_schedule[epoch],
                               dim=-1)
            t_ent = -(t_prob * torch.log(t_prob.clamp_min(1e-12))).sum(-1).mean()
            t_arg = t_prob.argmax(-1)
            proto_counts.index_add_(0, t_arg, torch.ones_like(t_arg))
            proto_prob_sum += t_prob.sum(0)

        # log statistics
        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1]) 
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)

        # student update
        optimizer.zero_grad(set_to_none=True)
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                utils.clip_gradients_foreach(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                utils.clip_gradients_foreach(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher (same math as param_k * m + (1 - m) * param_q, fused over all params)
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            torch._foreach_mul_(params_k, m)
            torch._foreach_add_(params_k, params_q, alpha=1 - m)

        # logging
        for key, value in [('loss', loss), ('acc', acc), ('t_ent', t_ent)] + list(all_loss.items()):
            pending.setdefault(key, []).append(value.detach().float())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        reached_max = args.max_iters and it + 1 >= args.max_iters
        if it_in_epoch % args.log_freq == 0 or it_in_epoch == n_iters - 1 or reached_max:
            flush_metrics()
        if reached_max:
            break
    flush_metrics()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # prototypes the teacher's argmax used this epoch, and the entropy of its epoch-mean distribution
    # (healthy: many prototypes and a high mean entropy while each t_ent is low)
    p_mean = proto_prob_sum / proto_prob_sum.sum()
    return_dict['t_protos'] = int((proto_counts > 0).sum())
    return_dict['t_mean_ent'] = float(-(p_mean * torch.log(p_mean.clamp_min(1e-12))).sum())
    print(f"Teacher CLS prototypes used: {return_dict['t_protos']} / {args.out_dim}, "
          f"entropy of mean target: {return_dict['t_mean_ent']:.3f} (max {math.log(args.out_dim):.3f})")
    return_dict['iters'] = it_in_epoch + 1

    return return_dict


class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, mim_start_epoch=0, compute_cls_loss_prob=1.0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mim_start_epoch = mim_start_epoch
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.compute_cls_loss_prob = compute_cls_loss_prob

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))

    def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
        
        compute_cls_loss = epoch < (self.mim_start_epoch + self.warmup_teacher_temp_epochs) \
            or np.random.rand() < self.compute_cls_loss_prob
        if student_local_cls is not None and compute_cls_loss:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        if compute_cls_loss:
            student_cls = student_cls / self.student_temp
            student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        if compute_cls_loss:
            teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
            teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_patch_c)):
            # [CLS] terms pair each teacher global view with every other student view, local crops included.
            # Patch outputs exist only for the global views: looping over student_patch_c (as in eb28c32) silently
            # dropped every local crop from the loss.
            for v in range(len(student_cls_c) if compute_cls_loss else len(student_patch_c)):
                if v == q:
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    mask = student_mask[v].flatten(1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                elif compute_cls_loss:
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
            
        if compute_cls_loss and n_loss_terms1 > 0:
            total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        else:
            total_loss1 = torch.tensor(0.0, device=student_patch.device)
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)
        self.update_center(teacher_cls, teacher_patch, compute_cls_loss)                  
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch, compute_cls_loss=True):
        """
        Update center used for teacher output.
        """
        if compute_cls_loss:
            cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
            dist.all_reduce(cls_center)
            cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
            self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)

class MinMaxScaling:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        min_tensor = tensor.min()
        max_tensor = tensor.max()
        if max_tensor == min_tensor:
            return torch.zeros_like(tensor)
        norm_tensor = (tensor - min_tensor) / (max_tensor - min_tensor)
        norm_tensor = norm_tensor * (self.max_val - self.min_val) + self.min_val
        return norm_tensor

class DataAugmentationiBOT(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            )
            # ),
            # T.RandomGrayscale(p=0.2),
        ])
        minmax_scaling = MinMaxScaling()
        normalize = T.Normalize(mean=[0.18713481659148107], std=[0.17456506099388783])
        permute = Permute([1, 0, 2, 3])
        # normalize = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = T.Compose([
            T.RandomResizedCrop(224, scale=global_crops_scale, interpolation=3),
            minmax_scaling,
            flip_and_color_jitter,
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.)),
            permute,
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = T.Compose([
            T.RandomResizedCrop(224, scale=global_crops_scale, interpolation=3),
            minmax_scaling,
            flip_and_color_jitter,
            T.RandomApply(
                [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.))],
                p=0.1
            ),
            T.RandomSolarize(threshold=0.5, p=0.2),
            permute,
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = T.Compose([
            T.RandomResizedCrop(96, scale=local_crops_scale, interpolation=3),
            minmax_scaling,
            flip_and_color_jitter,
            T.RandomApply(
                [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.))],
                p=0.5
            ),
            permute,
            normalize
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('iBOT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ibot(args)
