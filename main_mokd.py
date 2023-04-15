#!/usr/bin/env python
import argparse
import os
import sys
import builtins
import datetime
import time
import math
import json
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
# from torchvision import models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from utils import utils
from utils import checkpoint_io
from utils import optimizers
from augmentations.dino_augmentation import DINODataAugmentation
import backbones.vision_transformer as vits
import backbones.resnet as resnets
from models.mokd import *

method = "mokd"

def get_args_parser():
    parser = argparse.ArgumentParser(method, add_help=False)

    #################################
    #### input and output parameters ####
    #################################
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--experiment', default='exp', type=str, help='experiment name')

    #################################
    #### augmentation parameters ####
    #################################
    # multi-crop parameters
    parser.add_argument('--input_size', default=224, type=int, help='input image size')
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping.""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    #################################
    ####model parameters ####
    #################################
    parser.add_argument('--arch_cnn', default='resnet50', type=str,
        help="""Name of architecture to train. For quick experiments with ViTs""")
    parser.add_argument('--arch_vit', default='vit_small', type=str,
        help="""Name of architecture to train. For quick experiments with ViTs""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head..""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    # for ViTs
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    # for cross-distillation
    parser.add_argument("--lamda_c", default=1.0, type=float, help=""" weight for ct loss cnn.""")
    parser.add_argument("--lamda_t", default=1.0, type=float, help=""" weight for ct loss vit.""")

    #################################
    #### optim parameters ###
    #################################
    # training/pptimization parameters
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument("--lr_cnn", default=0.3, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training).""")
    parser.add_argument("--lr_vit", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). """)
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training.""")
    parser.add_argument('--clip_grad_cnn', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--clip_grad_vit', type=float, default=0.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    # temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs_cnn', default=50, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 50).')
    parser.add_argument('--warmup_teacher_temp_epochs_vit', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    #################################
    #### dist parameters ###
    #################################
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    return parser

def train(args):

    ######################## init dist ########################
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    ######################## preparing data ... ########################
    transform = DINODataAugmentation(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )

    dataset = datasets.ImageFolder(args.data_path, transform=transform)

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Loaded {len(dataset)} training images.")

    ######################## building networks ...########################
    if args.arch_vit in vits.__dict__.keys():
        student_vit = vits.__dict__[args.arch_vit](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher_vit = vits.__dict__[args.arch_vit](patch_size=args.patch_size)
        embed_dim_vit = student_vit.embed_dim
    else:
        print(f"Unknow architecture: {args.arch_vit}")

    if args.arch_cnn in resnets.__dict__.keys():
        student_cnn = resnets.__dict__[args.arch_cnn]()
        teacher_cnn = resnets.__dict__[args.arch_cnn]()
        embed_dim_cnn = student_cnn.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch_cnn}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    # use_bn = True if args.cnn_checkpoint is None else False
    student_cnn = CNNStudentWrapper(student_cnn, DINOHead(
        embed_dim_cnn,
        args.out_dim,
        use_bn=False,
        norm_last_layer=True),
        vits.THead_CNN(out_dim=args.out_dim, featuremap_size=args.input_size // 32, in_chans=embed_dim_cnn, 
        embed_dim=embed_dim_vit, depth=3, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)),
    )

    teacher_cnn = CNNTeacherWrapper(
        teacher_cnn,
        DINOHead(embed_dim_cnn, args.out_dim, False),
        vits.THead_CNN(out_dim=args.out_dim, featuremap_size=args.input_size // 32, in_chans=embed_dim_cnn, 
        embed_dim=embed_dim_vit, depth=3, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)),
        args.local_crops_number,
    )

    student_vit = ViTStudentWrapper(student_vit, DINOHead(
        embed_dim_vit,
        args.out_dim,
        use_bn=False,
        norm_last_layer=False),
        vits.THead_ViT(out_dim=args.out_dim, featuremap_size=14, 
        embed_dim=embed_dim_vit, depth=3, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)),
    )
    teacher_vit = ViTTeacherWrapper(
        teacher_vit,
        DINOHead(embed_dim_vit, args.out_dim, False),
        vits.THead_ViT(out_dim=args.out_dim, featuremap_size=14, 
        embed_dim=embed_dim_vit, depth=3, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)),
        args.local_crops_number,
    )

    # move networks to gpu
    student_cnn, teacher_cnn = student_cnn.cuda(), teacher_cnn.cuda()
    # synchronize batch norms
    if utils.has_batchnorms(student_cnn):
        student_cnn = nn.SyncBatchNorm.convert_sync_batchnorm(student_cnn)
        teacher_cnn = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_cnn)
        # use DDP wrapper to have synchro batch norms working...
        teacher_cnn = nn.parallel.DistributedDataParallel(teacher_cnn, device_ids=[args.gpu])
        teacher_cnn_without_ddp = teacher_cnn.module
    else:
        teacher_cnn_without_ddp = teacher_cnn
    student_cnn = nn.parallel.DistributedDataParallel(student_cnn, device_ids=[args.gpu], find_unused_parameters=False)
    teacher_cnn_without_ddp.load_state_dict(student_cnn.module.state_dict(), strict=False)
    for p in teacher_cnn.parameters():
        p.requires_grad = False
    print(f"CNN student and Teacher are built: they are both {args.arch_cnn} network.")

    student_vit, teacher_vit = student_vit.cuda(), teacher_vit.cuda()
    # synchronize batch norms
    if utils.has_batchnorms(student_vit):
        student_vit = nn.SyncBatchNorm.convert_sync_batchnorm(student_vit)
        teacher_vit = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_vit)
        # use DDP wrapper to have synchro batch norms working...
        teacher_vit = nn.parallel.DistributedDataParallel(teacher_vit, device_ids=[args.gpu])
        teacher_vit_without_ddp = teacher_vit.module
    else:
        teacher_vit_without_ddp = teacher_vit
    student_vit = nn.parallel.DistributedDataParallel(student_vit, device_ids=[args.gpu], find_unused_parameters=False)
    teacher_vit_without_ddp.load_state_dict(student_vit.module.state_dict(), strict=False)
    for p in teacher_vit.parameters():
        p.requires_grad = False
    print(f"ViT student and Teacher are built: they are both {args.arch_vit} network.")

    ######################## preparing loss ... ########################
    loss_cnn_fn = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs_cnn, # args.warmup_teacher_temp_epochs 50
        args.epochs,
    ).cuda()

    loss_vit_fn = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs_vit, # args.warmup_teacher_temp_epochs 30
        args.epochs,
    ).cuda()

    loss_cnn_ct_fn = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs_vit, # args.warmup_teacher_temp_epochs 30
        args.epochs,
    ).cuda()

    loss_vit_ct_fn = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs_vit, # args.warmup_teacher_temp_epochs 30
        args.epochs,
    ).cuda()

    loss_cnn_thead_fn = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs_cnn, # args.warmup_teacher_temp_epochs 50
        args.epochs,
    ).cuda()

    loss_vit_thead_fn = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs_vit, # args.warmup_teacher_temp_epochs 30
        args.epochs,
    ).cuda()

    loss_search_ct_fn = CTSearchLoss(
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs_vit,
        args.epochs,)

    ######################## preparing optimizer ... ########################
    params_groups_cnn = utils.get_params_groups2(student_cnn, head_name="transhead")
    if args.optimizer == "adamw":
        optimizer_cnn = torch.optim.AdamW(params_groups_cnn)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer_cnn = torch.optim.SGD(params_groups_cnn, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer_cnn = optimizers.LARS(params_groups_cnn)  # to use with convnet and large batches
    
    params_groups_vit = utils.get_params_groups(student_vit)
    optimizer_vit = torch.optim.AdamW(params_groups_vit)  # to use with ViTs

    # for mixed precision training
    fp16_scaler_cnn = None
    fp16_scaler_vit = None
    if args.use_fp16:
        fp16_scaler_cnn = torch.cuda.amp.GradScaler()
        fp16_scaler_vit = torch.cuda.amp.GradScaler()

    ######################## init schedulers ... ########################
    lr_schedule_cnn = utils.cosine_scheduler(
        args.lr_cnn * (args.batch_size_per_gpu * utils.get_world_size()) / 256., 
        0.0048, # args.min_lr
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule_cnn = utils.cosine_scheduler(
        1e-4, # args.weight_decay
        1e-4, # args.weight_decay_end
        args.epochs, len(data_loader),
    )

    lr_schedule_vit = utils.cosine_scheduler(
        args.lr_vit * (args.batch_size_per_gpu * utils.get_world_size()) / 256., 
        1e-5, # args.min_lr
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule_vit = utils.cosine_scheduler(
        0.04, # args.weight_decay
        0.4, # args.weight_decay_end
        args.epochs, len(data_loader),
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    summary_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 
    "tb", "{}_{}.{}_pretrain_{}".format(method, args.arch_cnn, args.arch_vit, args.experiment))) if args.rank == 0 else None

    to_restore = {"epoch": 0}
    checkpoint_io.restart_from_checkpoint(
        os.path.join(args.output_dir, "{}_{}.{}_pretrain_{}_temp.pth".format(method, args.arch_cnn, 
        args.arch_vit, args.experiment)),
        run_variables=to_restore,
        student_cnn=student_cnn,
        teacher_cnn=teacher_cnn,
        student_vit=student_vit,
        teacher_vit=teacher_vit,
        optimizer_cnn=optimizer_cnn,
        optimizer_vit=optimizer_vit,
        fp16_scaler_cnn=fp16_scaler_cnn,
        fp16_scaler_vit=fp16_scaler_vit,
        loss_cnn_fn=loss_cnn_fn,
        loss_vit_fn=loss_vit_fn,
        loss_cnn_ct_fn=loss_cnn_ct_fn,
        loss_vit_ct_fn=loss_vit_ct_fn,
        loss_cnn_thead_fn=loss_cnn_thead_fn,
        loss_vit_thead_fn=loss_vit_thead_fn,
    )
    start_epoch = to_restore["epoch"]

    ######################## start training ########################
    start_time = time.time()
    print("Starting {} training !".format(method))
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        ######################## training one epoch of DINO ... ########################
        train_stats = train_one_epoch(
            student_cnn, teacher_cnn, teacher_cnn_without_ddp, 
            student_vit, teacher_vit, teacher_vit_without_ddp, 
            loss_cnn_fn,  loss_vit_fn, loss_cnn_ct_fn, loss_vit_ct_fn, loss_cnn_thead_fn, loss_vit_thead_fn, loss_search_ct_fn,
            data_loader, optimizer_cnn, lr_schedule_cnn, wd_schedule_cnn, 
            optimizer_vit, lr_schedule_vit, wd_schedule_vit,
            momentum_schedule, epoch, 
            fp16_scaler_cnn, fp16_scaler_vit, 
            summary_writer, args)

        ########################writing logs ... ########################
        save_dict = {
            'student_cnn': student_cnn.state_dict(),
            'teacher_cnn': teacher_cnn.state_dict(),
            'student_vit': student_vit.state_dict(),
            'teacher_vit': teacher_vit.state_dict(),
            'optimizer_cnn': optimizer_cnn.state_dict(),
            'optimizer_vit': optimizer_vit.state_dict(),
            'epoch': epoch + 1,
            'arch_cnn': args.arch_cnn,
            'arch_vit': args.arch_vit,
            'loss_cnn_fn': loss_cnn_fn.state_dict(),
            'loss_vit_fn': loss_vit_fn.state_dict(),
            'loss_cnn_ct_fn': loss_cnn_ct_fn.state_dict(),
            'loss_vit_ct_fn': loss_vit_ct_fn.state_dict(),
            'loss_cnn_thead_fn': loss_cnn_thead_fn.state_dict(),
            'loss_vit_thead_fn': loss_vit_thead_fn.state_dict(),
        }
        if fp16_scaler_cnn is not None:
            save_dict['fp16_scaler_cnn'] = fp16_scaler_cnn.state_dict()
            save_dict['fp16_scaler_vit'] = fp16_scaler_vit.state_dict()

        utils.save_on_master(save_dict, os.path.join(args.output_dir, 
        "{}_{}.{}_pretrain_{}_temp.pth".format(method, args.arch_cnn, args.arch_vit, args.experiment)))

        if (args.saveckp_freq and epoch % args.saveckp_freq == 0) or epoch == args.epochs - 1:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 
            "{}_{}.{}_pretrain_{}_{:04d}.pth".format(method, args.arch_cnn, args.arch_vit, args.experiment, epoch)))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if utils.is_main_process():
            with (Path(args.output_dir) / "{}_{}.{}_pretrain_{}_log.txt".format(method, args.arch_cnn, args.arch_vit, args.experiment)).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    if args.rank == 0:
        summary_writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(student_cnn, teacher_cnn, teacher_cnn_without_ddp, 
            student_vit, teacher_vit, teacher_vit_without_ddp, 
            loss_cnn_fn,  loss_vit_fn, loss_cnn_ct_fn, loss_vit_ct_fn,  loss_cnn_thead_fn, loss_vit_thead_fn, loss_search_ct_fn,
            data_loader, optimizer_cnn, lr_schedule_cnn, wd_schedule_cnn, 
            optimizer_vit, lr_schedule_vit, wd_schedule_vit,
            momentum_schedule, epoch, 
            fp16_scaler_cnn, fp16_scaler_vit, 
            summary_writer, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    iters_per_epoch = len(data_loader)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer_cnn.param_groups):
            param_group["lr"] = lr_schedule_cnn[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule_cnn[it]
            if i == 2: # transhead
                param_group["lr"] = lr_schedule_vit[it]
                param_group["weight_decay"] = wd_schedule_vit[it]

        for i, param_group in enumerate(optimizer_vit.param_groups):
            param_group["lr"] = lr_schedule_vit[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule_vit[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        # student cnn
        with torch.cuda.amp.autocast(fp16_scaler_cnn is not None):
            s_cnn_m, s_cnn_t_g, s_cnn_out_g = student_cnn(images[:2])
            s_cnn_m_l, s_cnn_t_l, s_cnn_out_l = student_cnn(images[2:])
            s_cnn_m = torch.cat([s_cnn_m, s_cnn_m_l], dim=0)
            s_cnn_t = torch.cat([s_cnn_t_g, s_cnn_t_l], dim=0)

        # student vit
        with torch.cuda.amp.autocast(fp16_scaler_vit is not None):
            s_vit_m, s_vit_t_g,  s_vit_out_g = student_vit(images[:2])
            s_vit_m_l, s_vit_t_l,  s_vit_out_l = student_vit(images[2:])
            s_vit_m = torch.cat([s_vit_m, s_vit_m_l], dim=0)
            s_vit_t = torch.cat([s_vit_t_g, s_vit_t_l], dim=0)

        # teacher cnn
        with torch.cuda.amp.autocast(fp16_scaler_cnn is not None):
            t_cnn_m, t_cnn_t, t_cnn_search_l, _ = teacher_cnn(images[:2], local_token=s_vit_out_l.detach())
            
        # teacher vit
        with torch.cuda.amp.autocast(fp16_scaler_vit is not None):
            t_vit_m, t_vit_t, t_vit_search_l, _ = teacher_vit(images[:2], local_token=s_cnn_out_l.detach())

        # loss
        with torch.cuda.amp.autocast(fp16_scaler_cnn is not None):
            loss_cnn_m = loss_cnn_fn(s_cnn_m, t_cnn_m, epoch)
            loss_cnn_t = loss_cnn_thead_fn(s_cnn_t, t_cnn_t, epoch)
            loss_ct_cnn = loss_cnn_ct_fn(s_cnn_m, t_vit_m.detach(), epoch)
            loss_ct_search_cnn = loss_search_ct_fn(s_cnn_t_l, t_vit_search_l.detach(), epoch)

            loss_cnn_total = loss_cnn_m + loss_cnn_t + args.lamda_c * (loss_ct_cnn + loss_ct_search_cnn)

        with torch.cuda.amp.autocast(fp16_scaler_vit is not None):
            loss_vit_m = loss_vit_fn(s_vit_m, t_vit_m, epoch)
            loss_vit_t = loss_vit_thead_fn(s_vit_t, t_vit_t, epoch)       
            loss_ct_vit = loss_vit_ct_fn(s_vit_m, t_cnn_m.detach(), epoch)
            loss_ct_search_vit = loss_search_ct_fn(s_vit_t_l, t_cnn_search_l.detach(), epoch)
            
            loss_vit_total = loss_vit_m + loss_vit_t + args.lamda_t * (loss_ct_vit + loss_ct_search_vit)

        if not math.isfinite(loss_cnn_total.item()):
            print("Loss is {}, stopping training".format(loss_cnn_total.item()), force=True)
            sys.exit(1)
        if not math.isfinite(loss_vit_total.item()):
            print("Loss is {}, stopping training".format(loss_vit_total.item()), force=True)
            sys.exit(1)

        optimizer_cnn.zero_grad()
        param_norms = None
        if fp16_scaler_cnn is None:
            loss_cnn_total.backward()
            if args.clip_grad_cnn:
                param_norms = utils.clip_gradients(student_cnn, args.clip_grad_cnn)
            utils.cancel_gradients_last_layer(epoch, student_cnn, args.freeze_last_layer)
            optimizer_cnn.step()
        else:
            fp16_scaler_cnn.scale(loss_cnn_total).backward() # retain_graph=True
            if args.clip_grad_cnn:
                fp16_scaler_cnn.unscale_(optimizer_cnn)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student_cnn, args.clip_grad_cnn)
            utils.cancel_gradients_last_layer(epoch, student_cnn, args.freeze_last_layer)
            fp16_scaler_cnn.step(optimizer_cnn)
            fp16_scaler_cnn.update()

        # EMA update for the cnn teacher
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student_cnn.module.parameters(), teacher_cnn_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        optimizer_vit.zero_grad()
        param_norms = None
        if fp16_scaler_vit is None:
            loss_vit_total.backward()
            if args.clip_grad_vit:
                param_norms = utils.clip_gradients(student_vit, args.clip_grad_vit)
            utils.cancel_gradients_last_layer(epoch, student_vit, args.freeze_last_layer)
            optimizer_vit.step()
        else:
            fp16_scaler_vit.scale(loss_vit_total).backward()
            if args.clip_grad_vit:
                fp16_scaler_vit.unscale_(optimizer_vit)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student_vit, args.clip_grad_vit)
            utils.cancel_gradients_last_layer(epoch, student_vit, args.freeze_last_layer)
            fp16_scaler_vit.step(optimizer_vit)
            fp16_scaler_vit.update()

        # EMA update for the cnn teacher
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student_vit.module.parameters(), teacher_vit_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        if args.rank == 0:
            summary_writer.add_scalar("loss_cnn", loss_cnn_total.item(), it)
            summary_writer.add_scalar("loss_vit", loss_vit_total.item(), it)
            summary_writer.add_scalar("lr_cnn", optimizer_cnn.param_groups[0]["lr"], it)
            summary_writer.add_scalar("lr_vit", optimizer_vit.param_groups[0]["lr"], it)

        torch.cuda.synchronize()
        metric_logger.update(loss_cnn=loss_cnn_m.item())
        metric_logger.update(loss_vit=loss_vit_m.item())
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(method, parents=[get_args_parser()])
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train(args)
