# --------------------------------------------------------
# training code for CUT3R
# --------------------------------------------------------
# References:
# DUSt3R: https://github.com/naver/dust3r
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized
from itertools import islice

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import (
    PreTrainedModel,
    ARCroco3DStereo,
    ARCroco3DStereoConfig,
    inf,
    strip_module,
)  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa
from dust3r.viz import colorize
from dust3r.utils.render import get_render_results
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

import hydra
from omegaconf import OmegaConf
import logging
import pathlib
from tqdm import tqdm
import random
import builtins
import shutil

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
import torch.multiprocessing

from vggt.models.vggt import VGGT

torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")


def setup_for_distributed(accelerator: Accelerator):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (accelerator.num_processes > 8)
        if accelerator.is_main_process or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir):
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code", "{}".format(date_time))
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            ".vscode*",
            "assets*",
            "example*",
            "checkpoints*",
            "OLD*",
            "logs*",
            "out*",
            "runs*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
        ),
        dirs_exist_ok=True,
    )
    return dst_dir


def train(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_iter,
        mixed_precision="bf16",
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
        ],
    )
    device = accelerator.device

    setup_for_distributed(accelerator)

    printer.info("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        dst_dir = save_current_code(outdir=args.output_dir)
        printer.info(f"Saving current code to {dst_dir}")

    # auto resume
    if not args.resume:
        last_ckpt_fname = os.path.join(args.output_dir, f"checkpoint-last.pth")
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    printer.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    # fix the seed
    seed = args.seed + accelerator.state.process_index
    printer.info(
        f"Setting seed to {seed} for process {accelerator.state.process_index}"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.benchmark

    # training dataset and loader
    printer.info("Building train dataset %s", args.train_dataset)
    #  dataset and loader
    data_loader_train = build_dataset(
        args.train_dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=False,
        fixed_length=args.fixed_length
    )
    printer.info("Building test dataset %s", args.test_dataset)
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset,
            args.batch_size,
            args.num_workers,
            accelerator=accelerator,
            test=True,
            fixed_length=True
        )
        for dataset in args.test_dataset.split("+")
    }

    # model
    printer.info("Loading model")
    model = VGGT()

    # model: PreTrainedModel = eval(args.model)
    printer.info(f"All model parameters: {sum(p.numel() for p in model.parameters())}")


    printer.info(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)
    printer.info(
        f">> Creating test criterion = {args.test_criterion or args.train_criterion}"
    )
    test_criterion = eval(args.test_criterion or args.criterion).to(device)

    model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.long_context:
        model.fixed_input_length = False

    if args.pretrained and not args.resume:
        printer.info(f"Loading pretrained: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        printer.info(
            model.load_state_dict(ckpt, strict=True)
        )
        del ckpt  # in case it occupies memory

    # freeze
    printer.info("Freezing patch embedding and positional encoding parameters...")
    frozen_params = 0
    total_params = 0

    frozen_param_names = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        param.requires_grad = True

    if hasattr(model, 'aggregator') and hasattr(model.aggregator, 'patch_embed'):
        for param in model.aggregator.patch_embed.parameters():
            if param.requires_grad:
                param.requires_grad = False

    if hasattr(model, 'aggregator') and hasattr(model.aggregator, 'camera_token'):
        model.aggregator.camera_token.requires_grad = False

    if hasattr(model, 'aggregator') and hasattr(model.aggregator, 'register_token'):
        model.aggregator.register_token.requires_grad = False

    model.camera_head.requires_grad = False
    model.depth_head.requires_grad = False
    model.track_head.requires_grad = False

    


    for name, p in model.named_parameters():
        if not p.requires_grad:
            frozen_params += p.numel()
            frozen_param_names.append(name)

    printer.info(
        f"Frozen {frozen_params:,} parameters out of {total_params:,} total parameters. ({frozen_params / total_params:.2%})")
    printer.info(
        f"Trainable parameters: {total_params - frozen_params:,} ({(total_params - frozen_params) / total_params:.2%})")
    if frozen_param_names:
        printer.info(
            f"Example frozen parameters: {', '.join(frozen_param_names[:5])}{'...' if len(frozen_param_names) > 5 else ''}")



    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler(accelerator=accelerator)

    best_so_far = misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )
    if best_so_far is None:
        best_so_far = float("inf")

    accelerator.even_batches = False
    optimizer, model, data_loader_train = accelerator.prepare(
        optimizer, model, data_loader_train
    )

    def write_log_stats(epoch, train_stats, test_stats):
        if accelerator.is_main_process:
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far, data_iter_step):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            step=data_iter_step,
            fname=fname,
            best_so_far=best_so_far,
        )

    log_writer = (
        SummaryWriter(log_dir=args.output_dir) if accelerator.is_main_process else None
    )

    printer.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}

    for epoch in range(args.start_epoch, args.epochs + 1):

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if (
                args.save_freq
                and np.allclose(epoch / args.save_freq, int(epoch / args.save_freq))
                or epoch == args.epochs
            ):
                save_model(epoch - 1, "last", best_so_far, args.start_step)

        new_best = False
        # if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
        #     test_stats = {}
        #     for test_name, testset in data_loader_test.items():
        #         stats = test_one_epoch(
        #             model,
        #             teacher,
        #             test_criterion,
        #             testset,
        #             accelerator,
        #             device,
        #             epoch,
        #             log_writer=log_writer,
        #             args=args,
        #             prefix=test_name,
        #         )
        #         test_stats[test_name] = stats
        #
        #         # Save best of all
        #         if stats["loss_med"] < best_so_far:
        #             best_so_far = stats["loss_med"]
        #             new_best = True
        # # Save more stuff
        # write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far, args.start_step)
            if new_best:
                save_model(epoch - 1, "best", best_so_far, args.start_step)
        if epoch >= args.epochs:
            break  # exit after writing last test to disk


        # Train
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            accelerator,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args
        )


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    printer.info("Training time {}".format(total_time_str))

    save_final_model(accelerator, args, args.epochs, model, best_so_far=best_so_far)


def save_final_model(accelerator, args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": (
            model_without_ddp
            if isinstance(model_without_ddp, dict)
            else model_without_ddp.cpu().state_dict()
        ),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    printer.info(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(accelerator, to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, accelerator, test=False, fixed_length=False):
    split = ["Train", "Test"][test]
    printer.info(f"Building {split} Data loader for dataset: {dataset}")
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
        accelerator=accelerator,
        fixed_length=fixed_length
    )
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    def save_model(epoch, fname, best_so_far, data_iter_step):
        unwrapped_model = accelerator.unwrap_model(model)
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=unwrapped_model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            step=data_iter_step,
            fname=fname,
            best_so_far=best_so_far,
        )

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)


    optimizer.zero_grad()

    start_step = args.start_step

    data_iter = metric_logger.log_every(data_loader, args.print_freq, accelerator, header)

    for data_iter_step, batch in enumerate(data_iter):
            
        with accelerator.accumulate(model):
            # change the range of the image to [0, 1]
            if isinstance(batch, dict) and "img" in batch:
                batch["img"] = (batch["img"] + 1.0) / 2.0
            elif isinstance(batch, list) and all(isinstance(v, dict) and "img" in v for v in batch):
                for view in batch:
                    view["img"] = (view["img"] + 1.0) / 2.0

            epoch_f = epoch + data_iter_step / len(data_loader)
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                misc.adjust_learning_rate(optimizer, epoch_f, args)

            epoch_f = epoch + data_iter_step / len(data_loader)
            step = int(epoch_f * len(data_loader))

            result = loss_of_one_batch(
                batch,
                model,
                criterion,
                accelerator,
                inference=False,
                symmetrize_batch=False,
                use_amp=bool(args.amp),
            )
      
            loss, loss_details = result["loss"]  # criterion returns two values

            loss_value = float(loss)

            if not math.isfinite(loss_value):
                print(
                    f"Loss is {loss_value}, stopping training, loss details: {loss_details}"
                )
                sys.exit(1)
            if not result.get("already_backprop", False):
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=True,
                    clip_grad=1.0,
                )
                optimizer.zero_grad()

            is_metric = batch[0]["is_metric"]
            curr_num_view = len(batch)

            del loss
            tb_vis_img = (data_iter_step + 1) % accum_iter == 0 and (
                (step + 1) % (args.print_img_freq)
            ) == 0
            if not tb_vis_img:
                del batch
            else:
                torch.cuda.empty_cache()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
            metric_logger.update(step=step)
            #
            metric_logger.update(loss=loss_value, **loss_details)
            #
            if (data_iter_step + 1) % accum_iter == 0 and (
                (data_iter_step + 1) % (accum_iter * args.print_freq)
            ) == 0:
                loss_value_reduce = accelerator.gather(
                    torch.tensor(loss_value).to(accelerator.device)
                ).mean()  # MUST BE EXECUTED BY ALL NODES

                if log_writer is None:
                    continue
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(epoch_f * 1000)
                log_writer.add_scalar("train_loss", loss_value_reduce, step)
                log_writer.add_scalar("train_lr", lr, step)
                log_writer.add_scalar("train_iter", epoch_1000x, step)
                for name, val in loss_details.items():
                    if isinstance(val, torch.Tensor):
                        if val.ndim > 0:
                            continue
                    if isinstance(val, dict):
                        continue
                    log_writer.add_scalar("train_" + name, val, step)

            if tb_vis_img:
                if log_writer is None:
                    continue
                with torch.no_grad():
                    depths_cross, gt_depths_cross = get_render_results(
                        batch, result["pred"], self_view=False
                    )
                    for k in range(len(batch)):

                        loss_details[f"pred_depth_{k+1}"] = (
                            depths_cross[k].detach().cpu()
                        )
                        loss_details[f"gt_depth_{k+1}"] = (
                            gt_depths_cross[k].detach().cpu()
                        )

                imgs_stacked_dict = get_vis_imgs_new(
                    loss_details, args.num_imgs_vis, curr_num_view, is_metric=is_metric
                )
                for name, imgs_stacked in imgs_stacked_dict.items():
                    log_writer.add_images(
                        "train" + "/" + name, imgs_stacked, step, dataformats="HWC"
                    )
                del batch

        if (
            data_iter_step % int(args.save_freq * len(data_loader)) == 0
            and data_iter_step != 0
            and data_iter_step != len(data_loader) - 1
        ):
            print("saving at step", data_iter_step)
            save_model(epoch - 1, "last", float("inf"), data_iter_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    accelerator: Accelerator,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(0)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(0)

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        result = loss_of_one_batch(
            batch,
            model,
            criterion,
            accelerator,
            teacher=teacher,
            symmetrize_batch=False,
            use_amp=bool(args.amp),
        )

        loss_value, loss_details = result["loss"]  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

    printer.info("Averaged stats: %s", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            if isinstance(val, torch.Tensor):
                if val.ndim > 0:
                    continue
            if isinstance(val, dict):
                continue
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)


        depths_cross, gt_depths_cross = get_render_results(
            batch, result["pred"], self_view=False
        )
        for k in range(len(batch)):
            loss_details[f"pred_depth_{k+1}"] = depths_cross[k].detach().cpu()
            loss_details[f"gt_depth_{k+1}"] = gt_depths_cross[k].detach().cpu()

        imgs_stacked_dict = get_vis_imgs_new(
            loss_details,
            args.num_imgs_vis,
            args.num_test_views,
            is_metric=batch[0]["is_metric"],
        )
        for name, imgs_stacked in imgs_stacked_dict.items():
            log_writer.add_images(
                prefix + "/" + name, imgs_stacked, 1000 * epoch, dataformats="HWC"
            )

    del loss_details, loss_value, batch
    torch.cuda.empty_cache()

    return results


def batch_append(original_list, new_list):
    for sublist, new_item in zip(original_list, new_list):
        sublist.append(new_item)
    return original_list


def gen_mask_indicator(img_mask_list, ray_mask_list, num_views, h, w):
    output = []
    for img_mask, ray_mask in zip(img_mask_list, ray_mask_list):
        out = torch.zeros((h, w * num_views, 3))
        for i in range(num_views):
            if img_mask[i] and not ray_mask[i]:
                offset = 0
            elif not img_mask[i] and ray_mask[i]:
                offset = 1
            else:
                offset = 0.5
            out[:, i * w : (i + 1) * w] += offset
        output.append(out)
    return output


def vis_and_cat(
    gt_imgs,
    pred_imgs,
    cross_gt_depths,
    cross_pred_depths,
    cross_conf,
    ray_indicator,
    is_metric,
):
    cross_depth_gt_min = torch.quantile(cross_gt_depths, 0.01).item()
    cross_depth_gt_max = torch.quantile(cross_gt_depths, 0.99).item()
    cross_depth_pred_min = torch.quantile(cross_pred_depths, 0.01).item()
    cross_depth_pred_max = torch.quantile(cross_pred_depths, 0.99).item()
    cross_depth_min = min(cross_depth_gt_min, cross_depth_pred_min)
    cross_depth_max = max(cross_depth_gt_max, cross_depth_pred_max)

    cross_gt_depths_vis = colorize(
        cross_gt_depths,
        range=(
            (cross_depth_min, cross_depth_max)
            if is_metric
            else (cross_depth_gt_min, cross_depth_gt_max)
        ),
        append_cbar=True,
    )
    cross_pred_depths_vis = colorize(
        cross_pred_depths,
        range=(
            (cross_depth_min, cross_depth_max)
            if is_metric
            else (cross_depth_pred_min, cross_depth_pred_max)
        ),
        append_cbar=True,
    )


    if len(cross_conf) > 0:
        cross_conf_vis = colorize(cross_conf, append_cbar=True)

    gt_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    gt_imgs_vis[: gt_imgs.shape[0], : gt_imgs.shape[1]] = gt_imgs
    pred_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    pred_imgs_vis[: pred_imgs.shape[0], : pred_imgs.shape[1]] = pred_imgs
    ray_indicator_vis = torch.cat(
        [
            ray_indicator,
            torch.zeros(
                ray_indicator.shape[0],
                cross_pred_depths_vis.shape[1] - ray_indicator.shape[1],
                3,
            ),
        ],
        dim=1,
    )
    out = torch.cat(
        [
            ray_indicator_vis,
            gt_imgs_vis,
            pred_imgs_vis,
            cross_gt_depths_vis,
            cross_pred_depths_vis,
            cross_conf_vis,
        ],
        dim=0,
    )
    return out


def get_vis_imgs_new(loss_details, num_imgs_vis, num_views, is_metric):
    ret_dict = {}
    gt_img_list = [[] for _ in range(num_imgs_vis)]
    pred_img_list = [[] for _ in range(num_imgs_vis)]

    cross_gt_depth_list = [[] for _ in range(num_imgs_vis)]
    cross_pred_depth_list = [[] for _ in range(num_imgs_vis)]


    cross_view_conf_list = [[] for _ in range(num_imgs_vis)]
    cross_view_conf_exits = False

    img_mask_list = [[] for _ in range(num_imgs_vis)]
    ray_mask_list = [[] for _ in range(num_imgs_vis)]

    if num_views > 30:
        stride = 5
    elif num_views > 20:
        stride = 3
    elif num_views > 10:
        stride = 2
    else:
        stride = 1
    for i in range(0, num_views, stride):
        gt_imgs = 0.5 * (loss_details[f"gt_img{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
        width = gt_imgs.shape[2]
        pred_imgs = (
            0.5 * (loss_details[f"pred_rgb_{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
        )
        gt_img_list = batch_append(gt_img_list, gt_imgs.unbind(dim=0))
        pred_img_list = batch_append(pred_img_list, pred_imgs.unbind(dim=0))

        cross_pred_depths = (
            loss_details[f"pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        cross_gt_depths = (
            loss_details[f"gt_depth_{i+1}"]
            .to(gt_imgs.device)[:num_imgs_vis]
            .detach()
            .cpu()
        )
        cross_pred_depth_list = batch_append(
            cross_pred_depth_list, cross_pred_depths.unbind(dim=0)
        )
        cross_gt_depth_list = batch_append(
            cross_gt_depth_list, cross_gt_depths.unbind(dim=0)
        )

        if f"conf_{i+1}" in loss_details:
            cross_view_conf = loss_details[f"conf_{i+1}"][:num_imgs_vis].detach().cpu()
            cross_view_conf_list = batch_append(
                cross_view_conf_list, cross_view_conf.unbind(dim=0)
            )
            cross_view_conf_exits = True

        img_mask_list = batch_append(
            img_mask_list,
            loss_details[f"img_mask_{i+1}"][:num_imgs_vis].detach().cpu().unbind(dim=0),
        )
        ray_mask_list = batch_append(
            ray_mask_list,
            loss_details[f"ray_mask_{i+1}"][:num_imgs_vis].detach().cpu().unbind(dim=0),
        )

    # each element in the list is [H, num_views * W, (3)], the size of the list is num_imgs_vis
    gt_img_list = [torch.cat(sublist, dim=1) for sublist in gt_img_list]
    pred_img_list = [torch.cat(sublist, dim=1) for sublist in pred_img_list]
    cross_pred_depth_list = [
        torch.cat(sublist, dim=1) for sublist in cross_pred_depth_list
    ]
    cross_gt_depth_list = [torch.cat(sublist, dim=1) for sublist in cross_gt_depth_list]

    cross_view_conf_list = (
        [torch.cat(sublist, dim=1) for sublist in cross_view_conf_list]
        if cross_view_conf_exits
        else []
    )

    # each elment in the list is [num_views,], the size of the list is num_imgs_vis
    img_mask_list = [torch.stack(sublist, dim=0) for sublist in img_mask_list]
    ray_mask_list = [torch.stack(sublist, dim=0) for sublist in ray_mask_list]

    ray_indicator = gen_mask_indicator(
        img_mask_list, ray_mask_list, len(img_mask_list[0]), 30, width
    )

    for i in range(num_imgs_vis):
        out = vis_and_cat(
            gt_img_list[i],
            pred_img_list[i],
            cross_gt_depth_list[i],
            cross_pred_depth_list[i],
            cross_view_conf_list[i],
            ray_indicator[i],
            is_metric[i],
        )
        ret_dict[f"imgs_{i}"] = out
    return ret_dict


@hydra.main(
    version_base=None,
    config_path=str(os.path.dirname(os.path.abspath(__file__))) + "/../config",
    config_name="train.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    run()
