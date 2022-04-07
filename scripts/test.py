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

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# import cv2
# cv2.ocl.setUseOpenCL(False)

import av2.utils.io as io_utils
import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from mseg_semantic.utils.normalization_utils import get_imagenet_mean_std
from mseg_semantic.utils.avg_meter import SegmentationAverageMeter

import tbv.rendering_config as rendering_config
import tbv.training_config as training_config
import tbv.training.train_utils as train_utils
from tbv.rendering_config import BevRenderingConfig, EgoviewRenderingConfig, SensorViewpoint
from tbv.training_config import TrainingConfig
from tbv.training.train_utils import (
    BinaryClassificationAverageMeter,
    unnormalize_img,
)
from tbv.training.mcd_dataset import CLASSNAME_TO_CLASSIDX


def load_model_checkpoint(model: nn.Module, ckpt_fpath: str) -> nn.Module:
    """Load weights of a pre-trained Pytorch model into a Pytorch Module.

    Args:
        model:
        ckpt_fpath: file path to a model checkpoint, representing weights of a trained model.

    Returns:
        model: Pytorch model, with parameters loaded from the checkpoint file.
    """
    if Path(ckpt_fpath).exists():
        print(f"=> loading checkpoint '{ckpt_fpath}'")

        if torch.cuda.is_available():
            checkpoint = torch.load(ckpt_fpath)
        else:
            checkpoint = torch.load(ckpt_fpath, map_location=torch.device("cpu"))

        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"=> loaded checkpoint '{ckpt_fpath}'")
    else:
        raise RuntimeError(f"=> no checkpoint found at '{ckpt_fpath}'")

    return model


def run_test_epoch(
    args: TrainingConfig,
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    split: str,
    save_inference_viz: bool,
    run_gradcam: bool,
    ckpt_fpath: Path,
    eval_categories: List[str],
    filter_eval_by_visibility: bool,
) -> Dict[str, Any]:
    """

    Args:
        model:
        data_loader: data loader.
        split: dataset split.
        save_inference_viz: whether to save qualitative examples of correct and incorrect misclassifications.
        run_gradcam: whether to save GuidedGradCAM visualizations for each test example.
        ckpt_fpath: Path to pytorch pretrained model checkpoint.
        eval_categories:
        filter_eval_by_visibility: whether to evaluate only on visible nearby portions of the scene,
            or to evaluate on *all* nearby portions of the scene

    Returns:
        metrics_dict:
    """
    all_gts = np.zeros((0, 1))
    all_pred_dists = np.zeros((0, 1))

    sam = SegmentationAverageMeter()
    cam = BinaryClassificationAverageMeter()
    model.eval()

    # mapping from `(log_id, timestamp): pred_class`
    predictions_dict = {}

    for i, test_example in enumerate(data_loader):

        if args.loss_type in ["cross_entropy", "contrastive"]:
            x, xstar, labelmap, y, is_match, log_ids, timestamps = test_example

            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
                xstar = xstar.cuda(non_blocking=True)
                labelmap = labelmap.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                gt_is_match = is_match.cuda(non_blocking=True)
            else:
                gt_is_match = is_match

            if args.loss_type == "cross_entropy":

                if args.model_name == "EarlyFusionTwoHeadResnet":
                    is_match_probs, class_probs, loss = train_utils.cross_entropy_forward_two_head(
                        model, args, split, x, xstar, y, gt_is_match
                    )
                    probs = is_match_probs
                else:
                    probs, loss = train_utils.cross_entropy_forward(
                        model,
                        args,
                        split,
                        x,
                        xstar,
                        labelmap,
                        gt_is_match,
                        log_ids,
                        timestamps,
                        run_gradcam=run_gradcam,
                    )

                y_hat = torch.argmax(probs, dim=1)

                # MIN_CONF = 0.98
                # # if use mapping
                # if args.viewpoint == 'egoview':
                # 	timestamps = list(timestamps.cpu().numpy())
                # 	confs, _ = torch.max(probs.clone(), dim=1) # confidences
                # 	# y_hat is predicting `is_match`, so negate
                # 	change_detected_list = list(~(y_hat.clone().cpu().numpy().astype(bool)))
                # 	for j, (timestamp, log_id, change_detected) in enumerate(zip(timestamps, log_ids, change_detected_list)):

                # 		if change_detected != nearby_is_change and confs[j] > MIN_CONF:
                # 			print(f'w/ conf {confs[j]} Map overrode output @ {log_id} {timestamp}')
                # 			nearby_is_match = not nearby_is_change
                # 			y_hat[j] = nearby_is_match

                n = y.shape[0]
                sam.update_metrics_cpu(
                    pred=y_hat.cpu().numpy(),
                    target=gt_is_match.squeeze().cpu().numpy(),
                    num_classes=args.num_ce_classes,
                )

            elif args.loss_type == "contrastive":

                pred_dists, loss = model(x, xstar, gt_is_match)
                pred_is_match = pred_dists < args.contrastive_tp_dist_thresh

                # do some inference here, based on some thresholded distance
                verbose = True if (i % 10) == 0 else False
                if verbose:
                    avg_match_dist = torch.mean(pred_dists[gt_is_match == 1])
                    avg_mismatch_dist = torch.mean(pred_dists[gt_is_match == 0])
                    print(f"\tAvg match    dist={avg_match_dist.item():.2f}")
                    print(f"\tAvg mismatch dist={avg_mismatch_dist.item():.2f}")

                # positive should be when it is a match
                # negative should be when it is a mismatch
                cam.update(pred=pred_is_match, target=gt_is_match)

                all_gts = np.vstack([all_gts, gt_is_match.cpu().numpy().reshape(-1, 1)])
                all_pred_dists = np.vstack([all_pred_dists, pred_dists.cpu().numpy().reshape(-1, 1)])

            if save_inference_viz:
                if args.loss_type == "contrastive":
                    visualize_examples(
                        log_ids=log_ids,
                        batch_idx=i,
                        split=split,
                        x=x,
                        xstar=xstar,
                        labelmap=labelmap,
                        ckpt_fpath=ckpt_fpath,
                        eval_categories=eval_categories,
                        filter_eval_by_visibility=filter_eval_by_visibility,
                        gt_is_match=gt_is_match,
                        pred_dists=pred_dists,
                    )
                elif args.loss_type == "cross_entropy":
                    visualize_examples(
                        log_ids=log_ids,
                        batch_idx=i,
                        split=split,
                        x=x,
                        xstar=xstar,
                        labelmap=labelmap,
                        ckpt_fpath=ckpt_fpath,
                        eval_categories=eval_categories,
                        filter_eval_by_visibility=filter_eval_by_visibility,
                        y_hat=y_hat,
                        y_true=gt_is_match,
                        probs=probs,  # y,
                    )

            # visualize each misclassified example by hand

        elif args.loss_type in ["triplet"]:
            x_a, x_p, x_n = training_example
            # anchor, positive, negative moved to CUDA memory
            x_a = x_a.cuda(non_blocking=True)
            x_p = x_p.cuda(non_blocking=True)
            x_n = x_n.cuda(non_blocking=True)

            output, loss = model(x_a, x_p, x_n)

        if args.loss_type == "contrastive":
            avg_prec, avg_rec, avg_f1 = cam.get_metrics()
            print(
                f"\t{split} result at i [{i+1}/{len(data_loader)}]: prec/rec/F1 {avg_prec:.4f}/{avg_rec:.4f}/{avg_f1:.4f}"
            )

        elif args.loss_type == "cross_entropy":
            _, accs, _, avg_mAcc, _ = sam.get_metrics()
            print(
                f"\t{split} result at i [{i+1}/{len(data_loader)}]: mAcc{avg_mAcc:.4f}",
                "Cls Accuracies:",
                [float(f"{acc:.2f}") for acc in accs],
            )

    # TODO: serialize predictions to disk.
    io_utils.save_json_dict("predictions.json", predictions_dict)

    if args.loss_type == "cross_entropy":
        _, accs, _, avg_mAcc, _ = sam.get_metrics()
        print(f"{split} result: mAcc{avg_mAcc:.4f}", "Cls Accuracies:", [float(f"{acc:.2f}") for acc in accs])

        metrics_dict = {}

    elif args.loss_type == "contrastive":
        all_gts = all_gts.squeeze()
        all_pred_dists = all_pred_dists.squeeze()
        plot_pr_curve_sklearn(split, all_gts, all_pred_dists)
        plot_pr_curve_numpy(split, all_gts, all_pred_dists)

        avg_prec, avg_rec, avg_f1 = cam.get_metrics()
        print(f"Test result: prec/rec/F1 {avg_prec:.4f}/{avg_rec:.4f}/{avg_f1:.4f}")

        metrics_dict = {"recall": avg_rec, "precision": avg_prec, "f1": avg_f1}
    return metrics_dict


def visualize_examples(
    log_ids,
    batch_idx: int,
    split: str,
    x: torch.Tensor,
    xstar: torch.Tensor,
    labelmap: torch.Tensor,
    ckpt_fpath: Path,
    eval_categories: List[str],
    filter_eval_by_visibility: bool,
    **kwargs,
) -> None:
    """

    Args:
        log_ids:
        batch_idx:
        split: dataset split.
        x: tensor of shape (N,C,H,W) where C=3, representing a sensor image/data representation.
        xstar: tensor of shape (N,C,H,W) where C=3, representing map data representation.
        labelmap:
        ckpt_fpath: Path to pytorch pretrained model checkpoint.
        eval_categories
        filter_eval_by_categories: whether to evaluate only on visible nearby portions of the scene,
            or to evaluate on *all* nearby portions of the scene
        kwargs:
    """
    classidx_to_classname = {idx: name for name, idx in CLASSNAME_TO_CLASSIDX.items()}
    classidx_to_classname = {0: "mismatch", 1: "match"}

    if args.loss_type == "contrastive":
        gt_is_match = kwargs["gt_is_match"]
        pred_dists = kwargs["pred_dists"]
    elif args.loss_type == "cross_entropy":
        y_hat = kwargs["y_hat"]
        y_true = kwargs["y_true"]
        probs = kwargs["probs"]

    n, _, h, w = x.shape
    for j in range(n):

        # break
        # if j % 10 != 0:
        # 	continue

        mean, std = get_imagenet_mean_std()

        sensor_img = x[j, :, :, :]
        map_img = xstar[j, :, :, :]
        semantic_img = labelmap[j, :, :, :]

        log_id = log_ids[j]

        unnormalize_img(sensor_img, mean, std)
        unnormalize_img(map_img, mean, std)

        if args.viewpoint == SensorViewpoint.BEV:
            unnormalize_img(semantic_img, mean, std)
            semantic_img = semantic_img.cpu().numpy()
            semantic_img = np.transpose(semantic_img, (1, 2, 0)).astype(np.uint8)
        else:
            palette = np.array([[252, 233, 79], [206, 92, 0], [78, 154, 6], [92, 53, 102], [136, 138, 133]]).astype(
                np.uint8
            )
            rgb_semantic_img = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(5):
                rgb_semantic_img += palette[semantic_img[i].cpu().numpy().astype(np.uint8)]
            semantic_img = rgb_semantic_img

        sensor_img = sensor_img.cpu().numpy()
        map_img = map_img.cpu().numpy()

        sensor_img = np.transpose(sensor_img, (1, 2, 0)).astype(np.uint8)
        map_img = np.transpose(map_img, (1, 2, 0)).astype(np.uint8)

        fig = plt.figure(dpi=200, facecolor="white")

        plot_triplet = True
        if plot_triplet:
            num_cols = 3
        else:
            num_cols = 2

        fig.add_subplot(f"1{num_cols}1")
        plt.imshow(sensor_img)
        plt.axis("off")

        fig.add_subplot(f"1{num_cols}2")
        plt.title(str(map_img.shape))
        plt.imshow(map_img)
        plt.axis("off")

        if plot_triplet:
            fig.add_subplot(f"1{num_cols}3")
            plt.title(str(semantic_img.shape))
            plt.imshow(semantic_img)
            plt.axis("off")

        if args.loss_type == "contrastive":
            title = f"Is match: {gt_is_match[j].cpu().numpy().item() == 1}, "
            title += f"Distance: {pred_dists[j].item():.2f}"
        elif args.loss_type == "cross_entropy":
            pred_label_idx = y_hat[j].cpu().numpy().item()
            true_label_idx = y_true[j].cpu().numpy().item()
            title = f"{log_id[:4]} Pred Class: {classidx_to_classname[pred_label_idx]}"
            title += f", GT Class: {classidx_to_classname[true_label_idx]}"
            title += f"w prob {probs[j, pred_label_idx].cpu().numpy():.2f}"

        plt.suptitle(title)

        vis_save_dir = f"{Path(ckpt_fpath).parent}/{split}_set_examples_{eval_categories}_EvalByVisibility{filter_eval_by_visibility}"

        vis_save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(f"{vis_save_dir}/{log_id[:8]}_batch{batch_idx}_example{j}.jpg")
        # plt.show()
        plt.close("all")


# class MeanAveragePrecisionAvgMeter:
# 	def __init__(self):
# 		""" """
# 		pass

# 	def update(self, probs, y_true):
# 		""" """


# 	def get_metrics(self):
# 		""" """
# 		ninst = all_gts.size
# 		ranks = all_probs.argsort()[::-1]

# 		# rank the predictions and gt, from highest confidence to lowest
# 		all_pred_dists = all_pred_dists[ranks]
# 		all_gts = all_gts[ranks]

# 		avg_precision, precisions_interp =


def evaluate_model(
    model_args: TrainingConfig,
    dataset_args: Union[BevRenderingConfig, EgoviewRenderingConfig],
    ckpt_fpath: Path,
    split: str,
    save_inference_viz: bool,
    run_gradcam: bool,
    filter_eval_by_visibility: bool,
    eval_categories: List[str],
    save_gt_viz: bool,
) -> None:
    """Evaluate a pretrained model.

    Args:
        model_args: config containing information about trained model.
        dataset_args: config containing information about train/test dataset.
        ckpt_fpath: file path to a model checkpoint, representing weights of a trained model.
        split: dataset split.
        save_inference_viz: whether to save qualitative examples of correct and incorrect misclassifications.
        run_gradcam: whether to save GuidedGradCAM visualizations for each test example.
        filter_eval_by_visibility: whether to evaluate only on visible nearby portions of the scene,
           or to evaluate on *all* nearby portions of the scene.
        eval_categories: categories to consider when selecting frames to evaluate on.
        save_gt_viz: whether to save side-by-side visualizations of data examples,
            i.e. an image with horizontally stacked (sensor image, map image, blended combination).
            Will be classified by GT.
    """
    cudnn.benchmark = True

    data_loader = train_utils.get_dataloader(
        dataset_args=dataset_args,
        training_args=model_args,
        split=split,
        eval_categories=eval_categories,
        filter_eval_by_visibility=filter_eval_by_visibility,
        save_visualizations=save_gt_viz,
    )
    if len(data_loader) == 0:
        raise RuntimeError("Dataloader has length zero, meaning no data was found for inference.")

    model = train_utils.get_model(args=model_args, viewpoint=dataset_args.viewpoint)
    model = load_model_checkpoint(model, ckpt_fpath)

    with torch.no_grad():
        metrics_dict = run_test_epoch(
            args=model_args,
            model=model,
            data_loader=data_loader,
            split=split,
            save_inference_viz=save_inference_viz,
            run_gradcam=run_gradcam,
            ckpt_fpath=ckpt_fpath,
            eval_categories=eval_categories,
            filter_eval_by_visibility=filter_eval_by_visibility,
        )


def threshold_changes_by_prob(y_hat, probs, threshold: float) -> int:
    """Class 1 requires sufficient evidence to be selected

    Assume classes are zero and one only
    """
    n = probs.shape[0]
    chosen_probs = probs[torch.arange(n), y_hat]
    logicals = torch.logical_and(chosen_probs < threshold, y_hat == 1)
    y_hat = y_hat.bool()
    y_hat[logicals] = 0
    return y_hat.int()


def test_threshold_changes_by_prob() -> None:
    """ """
    probs = np.array(
        [
            [0.1, 0.91],  # 0.9 is high enough, remains class 1
            [0.49, 0.51],  # 0.51 is too low, clipped down to class 0
            [0.69, 0.72],  # clipped down to 0
            [0.5, 3.0],  # remains at 1
            [1, 0],  # stays at 0
            [0.5, 0.25],  # stays at 0
        ]
    ).reshape(6, 2)
    probs = torch.from_numpy(probs).cuda()

    y_hat = torch.argmax(probs, dim=1)

    threshold = 0.9
    y_hat_updated = threshold_changes_by_prob(y_hat, probs, threshold)
    y_hat_gt = torch.from_numpy(np.array([1, 0, 0, 1, 0, 0]))
    pdb.set_trace()
    assert np.allclose(y_hat_updated, y_hat_gt)


@click.command(help="Run map change detection inference on a split of TbV, using a pretrained model.")
@click.option(
    "--rendering_config_name",
    required=True,
    help="Name of rendering config file.",
    type=str,
)
@click.option("--training_config_name", required=True, type=str, help="name of training config file.")
@click.option("--gpu_ids", type=str, required=True, help="gpu IDs e.g. 0,1,2,3")
@click.option(
    "--num_workers",
    type=int,
    default=4,
    help="How many workers to use for dataloading during evaluation (overrides config via command line override)",
)
@click.option(
    "--save_inference_viz",
    default=False,
    help="Whether to save test set visualizations, defaults to false",
    type=bool,
)
@click.option(
    "--split",
    type=click.Choice(["val", "test"], case_sensitive=True),
    default="test",
    help="Dataset split to evaluate on.",
)
@click.option(
    "--ckpt_fpath", required=True, help="Path to pytorch pretrained model checkpoint.", type=click.Path(exists=True)
)
@click.option(
    "--filter_eval_by_visibility",
    type=bool,
    default=False,
    help="whether to evaluate only on visible nearby portions of the scene, or to evaluate on *all* nearby portions of the scene."
)
@click.option(
    "--run_gradcam",
    type=bool,
    default=False,
    help="whether to execute GradCAM and save visualizations of the salient regions."
)
@click.option(
    "--save_gt_viz",
    type=bool,
    default=False,
    help="whether to save side-by-side visualizations of data examples (classified by GT)," \
        "i.e. an image with horizontally stacked (sensor image, map image, blended combination)."
)
def run_test(
    rendering_config_name: str,
    training_config_name: str,
    gpu_ids: str,
    num_workers: int,
    save_inference_viz: bool,
    split: str,
    ckpt_fpath: str,
    filter_eval_by_visibility: bool,
    run_gradcam: bool,
    save_gt_viz: bool
) -> None:
    """Click entry point for model inference on a TbV data split."""

    # test_plot_pr_curve_numpy()

    # # dropout either map or semantics, 100% prob? uintuitive experiment...
    # config_path = (
    #     "checkpoints/2021_03_01_19_06_29/train_2021_03_01_egoview_w_labelmap_config_earlyfusion_dropout_v1.yaml"
    # )

    # opts.training_config_name = "2021_09_09_train_bev.yaml"
    # opts.rendering_config_name = "render_2021_09_04_bev_synthetic_config_t5820.yaml"

    # opts.training_config_name = "train_2021_03_01_egoview_w_labelmap_config_earlyfusion_dropout_v1.yaml"
    # opts.rendering_config_name = "render_2021_01_04_egoview.yaml"

    # load the train/test config for this model
    model_args = training_config.load_training_config(training_config_name)
    dataset_args = rendering_config.load_rendering_config(rendering_config_name)

    model_args.workers = num_workers

    print(model_args)
    print(dataset_args)

    # ckpt_fpath = "/home/jlambert/Documents/hd-map-change-detection/checkpoints/2021_03_01_19_06_29/train_ckpt.pth"
    # train_results_fpath = "/home/jlambert/Documents/hd-map-change-detection/checkpoints/2021_03_01_19_06_29/results-2021_03_01_19_06_29-train_2021_03_01_egoview_w_labelmap_config_earlyfusion_dropout_v1.json"

    eval_categories = ["change_lane_marking_color", "insert_crosswalk", "delete_crosswalk"]
    # args.eval_categories = ['change_lane_marking_color']
    # args.eval_categories = ['insert_crosswalk', 'delete_crosswalk']
    # args.eval_categories = ['delete_crosswalk']
    # args.eval_categories = ['insert_crosswalk']

    # from vis_training_progress import plot_metrics
    # plot_metrics(json_fpath=train_results_fpath, loss_type="cross_entropy")

    if torch.cuda.is_available():
        print(f"Using gpus {gpu_ids}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    evaluate_model(
        model_args=model_args,
        dataset_args=dataset_args,
        ckpt_fpath=Path(ckpt_fpath),
        split=split,
        save_inference_viz=save_inference_viz,
        run_gradcam=run_gradcam,
        filter_eval_by_visibility=filter_eval_by_visibility,
        eval_categories=eval_categories,
        save_gt_viz=save_gt_viz
    )

    # test_threshold_changes_by_prob()

    train_results_fpath = None
    if train_results_fpath is None:
        return

    train_results_json = io_utils.read_json_file(train_results_fpath)
    val_mAccs = train_results_json["val_mAcc"]
    print("Num epochs trained: ", len(val_mAccs))
    print("Max val mAcc: ", max(val_mAccs))
    print("Filter by visibility:", filter_eval_by_visibility)


if __name__ == "__main__":
    run_test()
