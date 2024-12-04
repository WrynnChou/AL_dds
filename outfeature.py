#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import os
import random
import shutil
import time
import warnings

import sklearn.decomposition
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.decomposition import PCA

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
         "batch size of all GPUs on the current node when "
         "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=15,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[60, 80],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by a ratio)",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--subsetsize", default=3000, type=int, help="Size of initial selected subset")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0,
    type=float,
    metavar="W",
    help="weight decay (default: 0.)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    default=False,
    help="Use multi-processing distributed training to launch "
         "N processes per node, which has N GPUs. This is the "
         "fastest way to use PyTorch for either single node or "
         "multi node data parallel training",
)
parser.add_argument(
    "--pretrained", default="", type=str, help="path to moco pretrained checkpoint"
)
parser.add_argument(
    "--active-indices", default="", type=str,
    help="path to samples whose indices are selected by active learning methods."
)
parser.add_argument(
    "--learning-mode", default="full", type=str,
    help="Decide which modes should be used for training;     'full' represents training models with full dataset, 'active' represents training models with initial dataset selected by active learning, 'random' represents training models with randomly selected initial labeled dataset."
)
parser.add_argument(
    "--feature_dir", default="", type=str,
    help="out feature dir path"
)
parser.add_argument(
    "--pca", default=False, type=bool,
    help="Weather do pca to reduce the dimension. Note that if not, the txt file might too large to read for R."
)
parser.add_argument(
    "--reduced_dim", default=16, type=int,
    help="pca dim"
)
best_acc1 = 0


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = 0

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=200)
    new_fc = torch.nn.Sequential(torch.nn.Linear(2048, 2048),
                                 torch.nn.Linear(2048, 16))
    model.fc = new_fc
    # freeze all layers but the last fc, remove if update all model parameters
    ####################################################
    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False
    ####################################################


    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            # rename moco pre-trained keys
            state_dict = checkpoint["state_dict"]
            state_dict['encoder_k.fc.1.weight'] = state_dict['encoder_k.fc.2.weight']
            state_dict['encoder_k.fc.1.bias'] = state_dict['encoder_k.fc.2.bias']
            state_dict['encoder_q.fc.1.weight'] = state_dict['encoder_q.fc.2.weight']
            state_dict['encoder_q.fc.1.bias'] = state_dict['encoder_q.fc.2.bias']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                # if k.startswith("encoder_q") and not k.startswith("encoder_q.fc.2")\
                #         and not k.startswith("encoder_q.fc.4"):
                if k.startswith("encoder_q"):
                    # remove prefix
                    state_dict[k[len("encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    else:
        print("Error: no pretrain model !!!")


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    # Train model with full dataset, randomly selected subset or active learning subset

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )


    out(train_loader, model, args)



def out(train_loader, model, args):

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in cifar100 mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    res = None
    label = None
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        if label is None:
            label = target
        else:
            label = torch.concatenate([label, target])
        # compute output
        output = model(images)
        if res is None:
            res = output
        else:
            res = torch.concatenate([res, output])
    res = res.cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    if args.pca:
        pca = PCA(n_components=args.reduced_dim)
        pc = pca.fit_transform(res)
        # pca.explained_variance_ratio_
        pathname = args.feature_dir + "/" + args.arch + "_" + str(args.reduced_dim) + "_feature_pca.txt"
        np.savetxt(pathname, pc)
    else:
        pathname = args.feature_dir + "/" + args.arch + "_feature.txt"
        np.savetxt(pathname, res)
    import torchvision
    train_data = torchvision.datasets.CIFAR10(
        root='data/cifar10/',  # 保存或者提取位置
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=False,  # 没下载就下载, 下载了就不用再下了
    )

if __name__ == "__main__":
    main()
