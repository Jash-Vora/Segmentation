#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   train.py
@Time    :   8/4/19 3:36 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import json
import timeit
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils import data
from torch.utils.checkpoint import checkpoint

import networks
import utils.schp as schp
from datasets.datasets import LIPDataSet
from datasets.target_generation import generate_edge_tensor
from utils.transforms import BGR2RGB_transform
from utils.criterion import CriterionAll
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.warmup_scheduler import SGDRScheduler


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='./data/LIP')
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=str, default='473,473')
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Training Strategy
    parser.add_argument("--learning-rate", type=float, default=7e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gpu", type=str, default='0,1,2')
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-epochs", type=int, default=10)
    parser.add_argument("--imagenet-pretrain", type=str, default='./pretrain_model/resnet101-imagenet.pth')
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", type=str, default='./log/checkpoint.pth.tar')
    parser.add_argument("--schp-start", type=int, default=100, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=10, help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str, default='./log/schp_checkpoint.pth.tar')
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    return parser.parse_args()


def main():
    args = get_arguments()
    print(args)

    start_epoch = 0
    cycle_n = 0

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'args.json'), 'w') as opt_file:
        json.dump(vars(args), opt_file)

    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    input_size = list(map(int, args.input_size.split(',')))

    cudnn.enabled = True
    cudnn.benchmark = True

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Model Initialization
    AugmentCE2P = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain)
    model = DataParallelModel(AugmentCE2P)
    model.cuda()

    IMAGE_MEAN = AugmentCE2P.mean
    IMAGE_STD = AugmentCE2P.std
    INPUT_SPACE = AugmentCE2P.input_space
    print(f'image mean: {IMAGE_MEAN}, std: {IMAGE_STD}, input space: {INPUT_SPACE}')

    if os.path.exists(args.model_restore):
        print(f'Resuming training from {args.model_restore}')
        checkpoint = torch.load(args.model_restore)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    SCHP_AugmentCE2P = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain)
    schp_model = DataParallelModel(SCHP_AugmentCE2P)
    schp_model.cuda()

    if os.path.exists(args.schp_restore):
        print(f'Resuming SCHP checkpoint from {args.schp_restore}')
        schp_checkpoint = torch.load(args.schp_restore)
        schp_model.load_state_dict(schp_checkpoint['state_dict'])
        cycle_n = schp_checkpoint['cycle_n']

    # Loss Function
    criterion = CriterionAll(lambda_1=args.lambda_s, lambda_2=args.lambda_e, lambda_3=args.lambda_c,
                             num_classes=args.num_classes)
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    # Data Loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])
    train_dataset = LIPDataSet(args.data_dir, 'train', crop_size=input_size, transform=transform)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size * len(gpus),
                                   num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    # Optimizer Initialization
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    lr_scheduler = SGDRScheduler(optimizer, total_epoch=args.epochs,
                                  eta_min=args.learning_rate / 100, warmup_epoch=10,
                                  start_cyclical=args.schp_start, cyclical_base_lr=args.learning_rate / 2,
                                  cyclical_epoch=args.cycle_epochs)

    total_iters = args.epochs * len(train_loader)
    start = timeit.default_timer()

    for epoch in range(start_epoch, args.epochs):
        lr_scheduler.step(epoch=epoch)
        lr = lr_scheduler.get_lr()[0]

        model.train()
        for i_iter, batch in enumerate(train_loader):
            images, labels, _ = batch
            images = images.cuda(non_blocking=True).float()
            labels = labels.cuda(non_blocking=True).long()

            edges = generate_edge_tensor(labels)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision training
            with torch.cuda.amp.autocast():
                preds = model(images)
                loss = criterion(preds, [labels, edges], cycle_n)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i_iter % 100 == 0:
                print(f'iter = {i_iter} of {total_iters}, lr = {lr}, loss = {loss.item()}')

        if (epoch + 1) % args.eval_epochs == 0:
            schp.save_schp_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, False, args.log_dir, filename=f'checkpoint_{epoch + 1}.pth.tar')

        # Self-Correction Cycle
        if (epoch + 1) >= args.schp_start and (epoch + 1 - args.schp_start) % args.cycle_epochs == 0:
            print(f'Self-correction cycle number {cycle_n}')
            schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
            cycle_n += 1
            schp.bn_re_estimate(train_loader, schp_model)
            schp.save_schp_checkpoint({
                'state_dict': schp_model.state_dict(),
                'cycle_n': cycle_n,
            }, False, args.log_dir, filename=f'schp_{cycle_n}_checkpoint.pth.tar')

        torch.cuda.empty_cache()
        end = timeit.default_timer()
        print(f'epoch = {epoch}, completed using {(end - start) / (epoch - start_epoch + 1):.2f} s')

    end = timeit.default_timer()
    print(f'Training finished in {end - start:.2f} seconds')


if __name__ == '__main__':
    main()
