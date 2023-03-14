# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
import argparse
import importlib
import time
import logging
from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import models
import data
from args import parse_args
from utils.schedules import get_lr_policy, get_optimizer
from utils.logging import (
    save_checkpoint,
    create_subdirs,
    parse_configs_file,
    clone_results_to_latest_subdir,
)
from utils.semisup import get_semisup_dataloader
from utils.model import (
    get_layers,
    prepare_model,
    initialize_scaled_score,
    scale_rand_init,
    show_gradients,
    current_model_pruned_fraction,
    sanity_check_paramter_updates,
    snip_init,
)
import shapley_init

import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os


# TODO: update wrn, resnet models. Save both subnet and dense version.
# TODO: take care of BN, bias in pruning, support structured pruning
from models.layers import SubnetConv, SubnetLinear

def calculate_remaining_ratio(model):
    remain_weights= 0
    all_weights = 0
    for layer in model.modules():
        if isinstance(layer, SubnetLinear):
            all_weights += layer.weight.shape[0] * layer.weight.shape[1]
            remain_weights -= torch.sum(layer.popup_scores == 0).item()
            # for i1 in range(layer.weight.shape[0]):
            #     for i2 in range(layer.weight.shape[1]):
            #         if layer.popup_scores[i1][i2] == 0:
            #             remain_weights -= 1
        elif isinstance(layer, SubnetConv):
            all_weights += layer.weight.shape[0] * layer.weight.shape[1] * layer.weight.shape[2] * layer.weight.shape[3]
            remain_weights -= torch.sum(layer.popup_scores == 0).item()
            # for i1 in range(layer.weight.shape[0]):
            #     for i2 in range(layer.weight.shape[1]):
            #         for i3 in range(layer.weight.shape[2]):
            #             for i4 in range(layer.weight.shape[3]):
            #                 if layer.popup_scores[i1][i2][i3][i4] == 0:
            #                     remain_weights -= 1
    remain_weights += all_weights
    print(f'remaining ratio : {1.0 * remain_weights / all_weights}')
    return 1.0 * remain_weights / all_weights


def main():
    args = parse_args()
    parse_configs_file(args)

    # sanity checks
    # if args.exp_mode in ["prune", "finetune"] and not args.resume:
    #     assert args.source_net, "Provide checkpoint to prune/finetune"

    # create resutls dir (for logs, checkpoints, etc.)
    result_main_dir = os.path.join(Path(args.result_dir), args.exp_name, args.exp_mode)

    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = os.path.join(
            result_main_dir,
            "{}--k-{:.2f}_trainer-{}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                n + 1,
                args.k,
                args.trainer,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    else:
        os.makedirs(result_main_dir, exist_ok=True)
        result_sub_dir = os.path.join(
            result_main_dir,
            "1--k-{:.2f}_trainer-{}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                args.k,
                args.trainer,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    create_subdirs(result_sub_dir)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Select GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")


    # Create model
    cl, ll = get_layers(args.layer_type)
    if len(gpu_list) > 1:
        print("Using multiple GPUs")
        model = nn.DataParallel(
            models.__dict__[args.arch](
                cl, ll, args.init_type, num_classes=args.num_classes
            ),
            gpu_list,
        ).to(device)
    else:
        model = models.__dict__[args.arch](
            cl, ll, args.init_type, num_classes=args.num_classes
        ).to(device)


    logger.info(model)

    # Customize models for training/pruning/fine-tuning
    prepare_model(model, args)

    # Setup tensorboard writer
    writer = SummaryWriter(os.path.join(result_sub_dir, "tensorboard"))


    # Dataloader
    D = data.__dict__[args.dataset](args, normalize=args.normalize)
    train_loader, test_loader = D.data_loaders()

    # logger.info(args.dataset, D, len(train_loader.dataset), len(test_loader.dataset))

    # Semi-sup dataloader
    if args.is_semisup:
        logger.info("Using semi-supervised training")
        sm_loader = get_semisup_dataloader(args, D.tr_train)
    else:
        sm_loader = None

    # autograd
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
    logger.info([criterion, optimizer, lr_policy])

    # train & val method
    trainer = importlib.import_module(f"trainer.{args.trainer}").train
    val = getattr(importlib.import_module("utils.eval"), args.val_method)

    # Load source_net (if checkpoint provided). Only load the state_dict (required for pruning and fine-tuning)

    # print()
    # print()
    # print()
    # print(args.source_net)
    # print()
    # print()
    # print()

    if args.source_net:
        if os.path.isfile(args.source_net):
            logger.info("=> loading source model from '{}'".format(args.source_net))
            checkpoint = torch.load(args.source_net, map_location=device)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            logger.info("=> loaded checkpoint '{}'".format(args.source_net))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    from torch.nn.parameter import Parameter
    for layer in model.modules():
        if isinstance(layer, SubnetLinear):
            layer.popup_scores = torch.ones(
                (layer.weight.shape[0], layer.weight.shape[1]),
                requires_grad=False).to(device)
        elif isinstance(layer, SubnetConv):
            layer.popup_scores = torch.ones(
                (layer.weight.shape[0], layer.weight.shape[1],layer.weight.shape[2], layer.weight.shape[3]),
                requires_grad=False).to(device)



    assert not (args.source_net and args.resume), (
        "Incorrect setup: "
        "resume => required to resume a previous experiment (loads all parameters)|| "
        "source_net => required to start pruning/fine-tuning from a source model (only load state_dict)"
    )


    best_prec1 = 0

    # show_gradients(model)

    # if args.source_net:
    #     last_ckpt = checkpoint["state_dict"]
    # else:
    #     last_ckpt = copy.deepcopy(model.state_dict())


    if args.evaluate or args.exp_mode in ["prune", "finetune"]:
        p1, _ = val(model, device, test_loader, criterion, args, writer)
        logger.info(f"Validation accuracy {args.val_method} for source-net: {p1}")
        if args.evaluate:
            return


    if args.shapley_layerwise_finetuning:
        model.train()
        shapley_loader = D.shapley_loader()

        cnt = 0
        for module in model.modules():
            if isinstance(module, SubnetConv) or isinstance(module, SubnetLinear):
                print(module)
                cnt += 1
        print(f'total layers: {cnt}')


        load_from_layer = -1

        num_layers = cnt
        for idx in range(num_layers):
            if idx==0 or idx==num_layers-1:
                continue

            if load_from_layer>0 and idx<load_from_layer:
                continue
            if load_from_layer==idx:
                model = torch.load(f'tmp_models/model_after_finetuning_layer_{load_from_layer - 1}_device_{device}')
                for ii, layer in enumerate(model.modules()):
                    if isinstance(layer, SubnetLinear) or isinstance(layer, SubnetConv):
                        layer.popup_scores = torch.load(
                                   f'tmp_models/popup_scores_after_finetuning_layer_{num_layers - 1 - idx}_device_{device}_{ii}')


            protocol = shapley_init.protocol3_layer(model, criterion, shapley_loader, device, args, idx)
            target_ratio = 0.01
            times = 1
            ratio_per_time = 1 - target_ratio ** (1.0 / times)

            for i in range(times):
                model = protocol.run_taylor(ratio_per_time)

            #model = protocol.run_taylor(0.99)
                calculate_remaining_ratio(model)
                print(f'after pruning layer {idx}')
                p1, _ = val(model, device, test_loader, criterion, args, writer)

                lr_tmp=1e-4
                epochs = 3

                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_tmp,
                                             weight_decay=args.wd, )

                torch.save(model, f'tmp_models/model_before_finetuning_layer_{num_layers-1-idx}_device_{device}')

                for epoch in range(0, epochs):
                    # lr_policy(epoch)  # adjust learning rate

                    trainer(
                        model,
                        device,
                        train_loader,
                        sm_loader,
                        criterion,
                        optimizer,
                        epoch,
                        args,
                        writer,
                    )
                    print(f'finetuing at epoch {epoch}: ')
                    p1, _ = val(model, device, test_loader, criterion, args, writer)

                torch.save(model, f'tmp_models/model_after_finetuning_layer_{num_layers - 1 - idx}_device_{device}')

                for ii, layer in enumerate(model.modules()):
                    if isinstance(layer, SubnetLinear) or isinstance(layer, SubnetConv):
                        torch.save(layer.popup_scores, f'tmp_models/popup_scores_after_finetuning_layer_{num_layers - 1 - idx}_device_{device}_{ii}')

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd,)

    if args.shapley_blockwise_finetuning:
        model.train()
        shapley_loader = D.shapley_loader()

        cnt = 0
        for module in model.modules():
            if isinstance(module, SubnetConv) or isinstance(module, SubnetLinear):
                print(module)
                cnt += 1
        print(f'total layers: {cnt}')


        load_from_layer = -1
        num_layers = cnt

        num_blocks = 9
        block_st = [1,4,7,10,13,16,19,22,25]
        block_en = [4,7,10,13,16,19,22,25,28]

        # block_st = list(range(1,27))
        # block_en = list(range(2,28))
        # num_blocks = len(block_en)


        for idx in range(num_blocks):

            protocol = shapley_init.protocol3_block(model, criterion, shapley_loader, device, args, block_st[idx], block_en[idx])
            model = protocol.run_taylor(0.99)
            calculate_remaining_ratio(model)
            print(f'after pruning block {idx}')
            p1, _ = val(model, device, test_loader, criterion, args, writer)

            lr_tmp=1e-4
            epochs = 5

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_tmp,
                                         weight_decay=args.wd, )

            # torch.save(model, f'tmp_models/model_before_finetuning_layer_{num_layers-1-idx}_device_{device}')

            for epoch in range(0, epochs):
                # lr_policy(epoch)  # adjust learning rate

                trainer(
                    model,
                    device,
                    train_loader,
                    sm_loader,
                    criterion,
                    optimizer,
                    epoch,
                    args,
                    writer,
                )
                print(f'finetuing at epoch {epoch}: ')
                p1, _ = val(model, device, test_loader, criterion, args, writer)

            # torch.save(model, f'tmp_models/model_after_finetuning_layer_{num_layers - 1 - idx}_device_{device}')

            # for ii, layer in enumerate(model.modules()):
            #     if isinstance(layer, SubnetLinear) or isinstance(layer, SubnetConv):
            #         torch.save(layer.popup_scores, f'tmp_models/popup_scores_after_finetuning_layer_{num_layers - 1 - idx}_device_{device}_{ii}')

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd,)


    if args.shapley_init:
        model.train()

        if args.trainer == 'smooth':
            shapley_loader = D.shapley_loader_smooth()
        else:
            shapley_loader = D.shapley_loader()

        target_ratio = 0.03
        times = 1
        ratio_per_time = 1 - target_ratio ** (1.0/times)

        protocol = shapley_init.protocol3(model, criterion, shapley_loader, device, args)
        # protocol.get_weights_and_gradient_information()
        lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)

        for i in range(times):
            model = protocol.run_taylor(ratio_per_time, i, times)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, )

            print(f'before finetuing at step {i}: ')
            ratio = calculate_remaining_ratio(model)
            # if ratio <= target_ratio:
            #     break
            p1, _ = val(model, device, test_loader, criterion, args, writer)

            if args.trainer == 'smooth':
                torch.save(model, f'model_before_finetuning_step_{i}_3')

            epochs=0

            for epoch in range(0, epochs):
                lr_policy(epoch)  # adjust learning rate
                trainer(
                    model,
                    device,
                    train_loader,
                    sm_loader,
                    criterion,
                    optimizer,
                    epoch,
                    args,
                    writer,
                )
                print(f'finetuing at step {i}: at epoch {epoch}: ')
                p1, _ = val(model, device, test_loader, criterion, args, writer)
            if args.trainer == 'smooth':
                torch.save(model, f'model_after_finetuning_step_{i}_3')

            # print(f'after finetuing at step {i}: ')
            # calculate_remaining_ratio(model)
            # p1, _ = val(model, device, test_loader, criterion, args, writer)

    #
    if args.evaluate or args.exp_mode in ["prune", "finetune"]:
        val(model, device, test_loader, criterion, args, writer)

        if args.evaluate:
            return



    # Start training
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
    for epoch in range(args.start_epoch, args.epochs + args.warmup_epochs):
        lr_policy(epoch)  # adjust learning rate

        # train
        trainer(
            model,
            device,
            train_loader,
            sm_loader,
            criterion,
            optimizer,
            epoch,
            args,
            writer,
        )

        calculate_remaining_ratio(model)

        # evaluate on test set
        if args.val_method == "smooth":
            prec1, radii = val(
                model, device, test_loader, criterion, args, writer, epoch
            )
            logger.info(f"Epoch {epoch}, mean provable Radii  {radii}")
        if args.val_method == "mixtrain" and epoch < args.epochs-3:
            prec1 = 0.0
        else:
            # prec1, _ = val(model, device, test_loader, criterion, args, writer, epoch)
            criterion = nn.CrossEntropyLoss()
            _, prec1 = val(model, device, test_loader, criterion, args, writer, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args,
            result_dir=os.path.join(result_sub_dir, "checkpoint"),
            save_dense=args.save_dense,
        )

        logger.info(
            f"Epoch {epoch}, val-method {args.val_method}, validation accuracy {prec1}, best_prec {best_prec1}"
        )
        if args.exp_mode in ["prune", "finetune"]:
            logger.info(
                "Pruned model: {:.2f}%".format(
                    current_model_pruned_fraction(
                        model, os.path.join(result_sub_dir, "checkpoint"), verbose=False
                    )
                )
            )
        # clone results to latest subdir (sync after every epoch)
        # Latest_subdir: stores results from latest run of an experiment.
        clone_results_to_latest_subdir(
            result_sub_dir, os.path.join(result_main_dir, "latest_exp")
        )

        # # Check what parameters got updated in the current epoch.
        # sw, ss = sanity_check_paramter_updates(model, last_ckpt)
        # logger.info(
        #     f"Sanity check (exp-mode: {args.exp_mode}): Weight update - {sw}, Scores update - {ss}"
        # )

    current_model_pruned_fraction(
        model, os.path.join(result_sub_dir, "checkpoint"), verbose=True
    )


if __name__ == "__main__":
    main()
