"""
Copyright (c) 2021
Argo AI, LLC, All Rights Reserved.

Notice: All information contained herein is, and remains the property
of Argo AI. The intellectual and technical concepts contained herein
are proprietary to Argo AI, LLC and may be covered by U.S. and Foreign
Patents, patents in process, and are protected by trade secret or
copyright law. This work is licensed under a CC BY-NC-SA 4.0 
International License.

Originating Authors: John Lambert
"""

import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import argoverse.utils.datetime_utils as datetime_utils
import argoverse.utils.json_utils as json_utils
import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from mseg_semantic.utils.avg_meter import AverageMeter, SegmentationAverageMeter
from mseg_semantic.utils.normalization_utils import get_imagenet_mean_std
# from mseg_semantic.utils.training_utils import poly_learning_rate
from mseg.utils.dir_utils import check_mkdir

import tbv.rendering_config as rendering_config
import tbv.training_config as training_config
import tbv.training.train_utils as train_utils
from tbv.rendering_config import BevRenderingConfig, EgoviewRenderingConfig
from tbv.training.train_utils import (
    BinaryClassificationAverageMeter,
    cross_entropy_forward,
    cross_entropy_forward_two_head,
)


def setup_logging(home_dir: str) -> None:
    """ """
    date_str = datetime_utils.generate_datetime_string()
    log_output_fpath = f"{home_dir}/logging/orthoimagery_program_{date_str}.log"
    check_mkdir(f"{home_dir}/logging")
    print(f"Log will be saved to {log_output_fpath}")

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=log_output_fpath,
        level=logging.INFO,
    )
    logging.getLogger("boto").setLevel(logging.CRITICAL)
    logging.debug("Init Debug")
    logging.info("Init Info")
    logging.warning("Init Warning")
    logging.critical("Init Critical")


# HOME_DIR = '/home/jlambert'
HOME_DIR = "/home/ubuntu"
setup_logging(HOME_DIR)


def main(dataset_args: Union[BevRenderingConfig, EgoviewRenderingConfig], training_args: TrainingConfig) -> None:
    """We will load the train vs. validation splits from a file"""
    np.random.seed(0)
    random.seed(0)
    cudnn.benchmark = True

    logging.info(str(args))

    # compute_data_mean_std(args.data_root, args.modality, args.interp_type)
    # load the cnst vs. awesome tags

    train_loader = train_utils.get_dataloader(dataset_args=dataset_args, training_args=training_args, split="train")
    val_loader = train_utils.get_dataloader(dataset_args=dataset_args, training_args=training_args, split="val")

    model = train_utils.get_model(args=training_args, viewpoint=dataset_args.viewpoint)
    optimizer = train_utils.get_optimizer(args, model)

    cfg_stem = args.cfg_stem
    exp_start_time = datetime_utils.generate_datetime_string()

    results_dict = defaultdict(list)
    results_dict["args"] = [{k: v for k, v in args.items()}]

    for epoch in range(args.num_epochs):
        logging.info(f"On epoch {epoch}")
        train_metrics_dict = run_epoch(args, epoch, model, train_loader, optimizer, split="train")

        for k, v in train_metrics_dict.items():
            results_dict[f"train_{k}"] += [v]

        val_metrics_dict = run_epoch(args, epoch, model, val_loader, optimizer, split="val")

        for k, v in val_metrics_dict.items():
            results_dict[f"val_{k}"] += [v]

        # critical accuracy statistic
        crit_acc_stat = "val_"
        if args.loss_type == "contrastive":
            crit_acc_stat += "f1"
        elif args.loss_type == "cross_entropy":
            crit_acc_stat += "mAcc"
        else:
            raise RuntimeError("Undefined loss type")

        if epoch > 0:
            curr_stat = results_dict[crit_acc_stat][-1]
            prev_best = max(results_dict[crit_acc_stat][:-1])
            is_best = curr_stat > prev_best

        # IF THE BEST MODEL, SAVE IT TO DISK
        if epoch == 0 or is_best:

            results_dir = Path(args.model_save_dirpath) / exp_start_time
            results_dir.mkdir(parents=True, exist_ok=True)
            ckpt_fpath = results_dir / "train_ckpt.pth"
            logging.info(f"Saving checkpoint to: {ckpt_fpath}")

            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "max_epochs": args.num_epochs,
                },
                ckpt_fpath,
            )

        # save loss values to json for plotting (replacing the same file each time)
        results_json_fpath = f"{results_dir}/results-{exp_start_time}-{cfg_stem}.json"
        json_utils.save_json_dict(results_json_fpath, results_dict)
        shutil.copyfile(args.config_path, f"{results_dir}/{Path(args.config_path).name}")

        logging.info("Results on crit stat: " + str([f"{v:.3f}" for v in results_dict[crit_acc_stat]]))
        # this is the `patience' value
        if epoch < 5 or is_best:
            logging.info(f"Epoch {epoch} was <5 or the best so far")
            continue

        if args.lr_annealing_strategy == "reduce_on_plateau":
            # history already has length 6, and is not the best

            if not acc_improved_over_last_k_epochs(results_dict[crit_acc_stat], k=5):
                # Decay the learning rate if we cannot improve val. acc. over k epochs.
                for param_group in optimizer.param_groups:
                    logging.info("Learning rate was: ", param_group["lr"])
                    param_group["lr"] *= args.reduce_on_plateau_power
                    logging.info("After k strikes, learning rate decayed to: ", param_group["lr"])


def run_epoch(args, epoch: int, model, data_loader, optimizer, split: str):
    """Run all data belonging to a particular split through the network"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    cam = BinaryClassificationAverageMeter()

    sam = SegmentationAverageMeter()

    finegrained_sam = SegmentationAverageMeter()

    if split == "train":
        model.train()
    else:
        model.eval()
    end = time.time()

    for iter, training_example in enumerate(data_loader):

        if iter % 10 == 0:
            logging.info(f"\tOn iter {iter}")
        data_time.update(time.time() - end)
        # if args.zoom_factor != 8:
        # h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
        # w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)

        # Could to multi-task, with one triplet head, and one cross-entropy head

        if args.loss_type in ["cross_entropy", "contrastive"]:
            x, xstar, labelmap, y, is_match, log_ids, timestamps = training_example

            # for k in [5,10,15,20,25,30,35,40,45,50,55,60]:
            # 	import matplotlib.pyplot as plt
            # 	plt.figure(figsize=(10,5))
            # 	mean, std = get_imagenet_mean_std()
            # 	from test import unnormalize_img
            # 	unnormalize_img(x[k].cpu(), mean, std)
            # 	unnormalize_img(xstar[k].cpu(), mean, std)
            # 	unnormalize_img(labelmap[k].cpu(), mean, std)

            # 	plt.subplot(1,3,1)
            # 	plt.imshow(x[k].numpy().transpose(1,2,0).astype(np.uint8))
            # 	plt.subplot(1,3,2)
            # 	plt.imshow(xstar[k].numpy().transpose(1,2,0).astype(np.uint8))
            # 	plt.subplot(1,3,3)
            # 	plt.imshow(labelmap[k].numpy().transpose(1,2,0).astype(np.uint8))

            # 	plt.title(str(is_match[k].numpy().item()))

            # 	# plt.subplot(2,4,1)
            # 	# plt.imshow(x[k].numpy().transpose(1,2,0).astype(np.uint8))
            # 	# plt.subplot(2,4,2)
            # 	# plt.imshow(xstar[k].numpy().transpose(1,2,0).astype(np.uint8))
            # 	# plt.subplot(2,4,3)

            # 	# plt.imshow(labelmap[k,0].numpy().squeeze().astype(np.uint8))
            # 	# plt.subplot(2,4,4)
            # 	# plt.imshow(labelmap[k,1].numpy().squeeze().astype(np.uint8))
            # 	# plt.subplot(2,4,5)
            # 	# plt.imshow(labelmap[k,2].numpy().squeeze().astype(np.uint8))
            # 	# plt.subplot(2,4,6)
            # 	# plt.imshow(labelmap[k,3].numpy().squeeze().astype(np.uint8))
            # 	# plt.subplot(2,4,7)
            # 	# plt.imshow(labelmap[k,4].numpy().squeeze().astype(np.uint8))

            # 	plt.show()
            # 	plt.close('all')

            x = x.cuda(non_blocking=True)
            xstar = xstar.cuda(non_blocking=True)
            labelmap = labelmap.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            gt_is_match = is_match.cuda(non_blocking=True)

            n = x.size(0)

            if args.loss_type == "cross_entropy":

                if args.model_name == "EarlyFusionTwoHeadResnet":
                    is_match_probs, class_probs, loss = cross_entropy_forward_two_head(
                        model, args, split, x, xstar, y, gt_is_match
                    )
                    finegrained_sam.update_metrics_cpu(
                        pred=torch.argmax(class_probs, dim=1).cpu().numpy(),
                        target=y.squeeze().cpu().numpy(),
                        num_classes=args.num_finegrained_classes,
                    )

                else:
                    is_match_probs, loss = cross_entropy_forward(model, args, split, x, xstar, labelmap, gt_is_match)

                # num_classes = len(list(set(CLASSNAME_TO_CLASSIDX.values())))
                sam.update_metrics_cpu(
                    pred=torch.argmax(is_match_probs, dim=1).cpu().numpy(),
                    target=gt_is_match.squeeze().cpu().numpy(),
                    num_classes=args.num_ce_classes,
                )

            elif args.loss_type == "contrastive":

                if split == "train":
                    pred_dists, loss = model(x, xstar, gt_is_match)
                else:
                    # for some strange reason, eval() memory use is more than train()
                    # so use no_grad()
                    with torch.no_grad():
                        pred_dists, loss = model(x, xstar, gt_is_match)
                pred_is_match = pred_dists < args.contrastive_tp_dist_thresh

                # do some inference here, based on some thresholded distance
                verbose = True if (iter % 10) == 0 else False
                if verbose:
                    avg_match_dist = torch.mean(pred_dists[gt_is_match == 1])
                    avg_mismatch_dist = torch.mean(pred_dists[gt_is_match == 0])
                    logging.info(f"\tAvg match    dist={avg_match_dist.item():.2f}")
                    logging.info(f"\tAvg mismatch dist={avg_mismatch_dist.item():.2f}")

                # positive should be when it is a match
                # negative should be when it is a mismatch
                cam.update(pred=pred_is_match, target=gt_is_match)

        elif args.loss_type in ["triplet"]:
            x_a, x_p, x_n = training_example
            # anchor, positive, negative moved to CUDA memory
            x_a = x_a.cuda(non_blocking=True)
            x_p = x_p.cuda(non_blocking=True)
            x_n = x_n.cuda(non_blocking=True)

            n = x_a.size(0)

            output, loss = model(x_a, x_p, x_n)

        # if not args.multiprocessing_distributed:
        # 	main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)

        max_iter = args.num_epochs * len(data_loader)
        current_iter = epoch * len(data_loader) + iter + 1 + args.resume_iter

        if split == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if split == "train" and args.lr_annealing_strategy == "poly":
            # decay learning rate only during training
            current_lr = train_utils.poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.poly_lr_power)

            if iter % 10 == 0:
                logging.info(
                    f"\tLR:{current_lr:.5f}, base_lr: {args.base_lr:.3f}, current_iter:{current_iter}, max_iter:{max_iter}, power:{args.poly_lr_power}"
                )
                # logger.info(f'LR:{current_lr}, base_lr: {args.base_lr}, current_iter:{current_iter}, max_iter:{max_iter}, power:{args.power}')

            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        loss_meter.update(loss.item(), n)

        # may need to divide loss by n
        # may need to .detach() loss variable

        if iter > 0:
            # ignore the first iter, while GPU warms up
            batch_time.update(time.time() - end)
        end = time.time()

        if iter % 10 == 0 and args.loss_type == "contrastive":
            avg_prec, avg_rec, avg_f1 = cam.get_metrics()
            logging.info(
                f"\t{split} result at iter [{iter+1}/{len(data_loader)}]: prec/rec/F1 {avg_prec:.4f}/{avg_rec:.4f}/{avg_f1:.4f}"
            )
            train_utils.print_time_remaining(batch_time, current_iter, max_iter)

        elif iter % 10 == 0 and args.loss_type == "cross_entropy":

            _, accs, _, avg_mAcc, _ = sam.get_metrics()
            logging.info(f"\t{args.num_ce_classes}-Cls Accuracies:" + str([float(f"{acc:.2f}") for acc in accs]))
            # avg_mAcc = acc_meter.avg
            logging.info(
                f"\t{split} result at iter [{iter+1}/{len(data_loader)}]: {args.num_ce_classes}-CE mAcc {avg_mAcc:.4f}"
            )
            train_utils.print_time_remaining(batch_time, current_iter, max_iter)

            if args.model_name == "EarlyFusionTwoHeadResnet":
                _, accs, _, avg_mAcc, _ = finegrained_sam.get_metrics()
                logging.info(
                    f"\t{args.num_finegrained_classes}-Cls Accuracies:" + str([float(f"{acc:.2f}") for acc in accs])
                )
                logging.info(
                    f"\t{split} result at iter [{iter+1}/{len(data_loader)}]: {args.num_finegrained_classes}-CE mAcc {avg_mAcc:.4f}"
                )

    if args.loss_type == "contrastive":
        avg_prec, avg_rec, avg_f1 = cam.get_metrics()
        logging.info(
            f"{split} result at epoch [{epoch+1}/{args.num_epochs}]: prec/rec/F1 {avg_prec:.4f}/{avg_rec:.4f}/{avg_f1:.4f}"
        )
        metrics_dict = {"avg_loss": loss_meter.avg, "recall": avg_rec, "precision": avg_prec, "f1": avg_f1}
    elif args.loss_type == "cross_entropy":

        _, accs, _, avg_mAcc, _ = sam.get_metrics()
        logging.info(f"{split} result at epoch [{epoch+1}/{args.num_epochs}]: mAcc{avg_mAcc:.4f}")
        logging.info("Cls Accuracies:" + str([float(f"{acc:.2f}") for acc in accs]))

        metrics_dict = {
            "avg_loss": loss_meter.avg,
            "mAcc": avg_mAcc,
        }
    return metrics_dict


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config_path", type=str, required=True, help="path to training config")
    parser.add_argument("--gpu_ids", type=str, required=True, help="gpu IDs e.g. 0,1,2,3")
    opts = parser.parse_args()

    print(f"Using gpus {opts.gpu_ids}")
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

    opts.dataset_config_path = ""

    dataset_args = rendering_config.load_rendering_config(opts.dataset_config_path)
    training_args = training_config.load_training_config(opts.training_config_path)
    args.cfg_stem = Path(opts.config_path).stem
    args.config_path = opts.config_path
    main(dataset_args, training_args)
