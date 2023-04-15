#!/usr/bin/env python
import os
import sys
import argparse
import json
from pathlib import Path
import time
import builtins
import random

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as transforms
from torchvision import models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

from utils import utils
from utils import checkpoint_io
from utils import optimizers
from utils import metrics
import backbones.vision_transformer as vits

def main():
    parser = argparse.ArgumentParser("Linear Evaluation", parents=[get_args_parser()])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    train(args)


def get_args_parser():
    parser = argparse.ArgumentParser("Linear Evaluation", add_help=False)

    #################################
    #### input and output parameters ####
    #################################
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--pretrained_weights', default='', type=str, 
    help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, 
    help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--evaluate', dest='evaluate', type=str, default=None, help='evaluate model on validation set')
    parser.add_argument('--method', default='moco', type=str, help='model name')
    parser.add_argument('--experiment', default='exp', type=str, help='experiment name')

    #################################
    ####model parameters ####
    #################################
    parser.add_argument('--arch', default='resnet50', type=str, help='Architecture')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    #for ViTs
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. Use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=0, type=int,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.""")
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    
    #################################
    #### optim parameters ###
    #################################
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--optimizer', default='sgd', type=str,
        choices=['sgd', 'lars'], help="""Type of optimizer.""")
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--val_freq', default=5, type=int, help="Epoch frequency for validation.")

    #################################
    #### dist parameters ###
    #################################
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')

    return parser


def train(args):
     ######################## building network ...  ########################
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
        # embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()

    ######################## load pretrained weights to evaluate ########################
    is_load_success = checkpoint_io.load_pretrained_weights(model, args.pretrained_weights, 
    args.checkpoint_key, args.method)
    if is_load_success:
        print(f"Model {args.arch} built.")
    else:
        sys.exit(1)

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

     ######################## preparing data  ########################
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    ######################## just do evaluation ########################
    if args.evaluate != None:
        is_load_success = checkpoint_io.load_pretrained_linear_weights(linear_classifier, args.evaluate, args.method)
        if is_load_success == False:
            sys.exit(1)
        test_stats = validate(val_loader, model, linear_classifier, args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    ######################## preparing data ########################
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    ######################## preparing optimizer ########################
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0,
    )
    if args.optimizer == "lars":
        optimizer = optimizers.LARS(linear_classifier.parameters(),
         lr=args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., 
         momentum=0.9,
         weight_decay=0,
         )  # to use with convnet and large batches

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    ######################## optionally resume training ########################
    to_restore = {"epoch": 0, "best_acc": 0.}
    checkpoint_io.restart_from_checkpoint(
        os.path.join(args.output_dir, 
        "{}_{}_linear_{}.pth".format(args.method, args.arch, args.experiment)),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    summary_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 
    "tb", "{}_{}_linear_{}".format(args.method, args.arch, args.experiment))) if utils.is_main_process() else None

    ######################## start training ########################
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, linear_classifier, optimizer, train_loader, epoch, summary_writer, args)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},  'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate(val_loader, model, linear_classifier, args)
           
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            if utils.is_main_process():
                summary_writer.add_scalar("acc1", test_stats["acc1"], epoch)
                summary_writer.add_scalar("acc5", test_stats["acc5"], epoch)
       
        if utils.is_main_process():
            with (Path(args.output_dir) / "{}_{}_linear_{}_log.txt".format(args.method, args.arch, args.experiment)).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, 
            "{}_{}_linear_{}.pth".format(args.method, args.arch, args.experiment)))
    
    if utils.is_main_process():
        summary_writer.close()
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
    time.sleep(30)


def train_one_epoch(model, linear_classifier, optimizer, loader, epoch, summary_writer, args):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    iters_per_epoch = len(loader)
    for it, (inp, target) in enumerate(metric_logger.log_every(loader, 20, header)):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, args.n_last_blocks) # take n last block out
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if args.avgpool_patchtokens:
                    output = torch.cat((output.unsqueeze(-1), 
                    torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
                if len(output.shape) != 2:
                    output = output.squeeze()
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # logging
        if utils.is_main_process():
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + it)
            summary_writer.add_scalar("learning rate", optimizer.param_groups[0]["lr"], 
            epoch * iters_per_epoch + it)
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(val_loader, model, linear_classifier, args):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, args.n_last_blocks) # take n last block out
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if args.avgpool_patchtokens:
                    output = torch.cat((output.unsqueeze(-1), 
                    torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = metrics.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = metrics.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    main()
