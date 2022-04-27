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

from typing import Callable, List, Optional, Tuple

import cv2

cv2.ocl.setUseOpenCL(False)
from mseg_semantic.utils.avg_meter import AverageMeter, SegmentationAverageMeter
from mseg_semantic.utils.normalization_utils import get_imagenet_mean_std
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import tbv.utils.captum_utils as captum_utils
import tbv.utils.logger_utils as logger_utils
from tbv.models.early_fusion import EarlyFusionCEResnet, EarlyFusionCEResnetWLabelMap, EarlyFusionTwoHeadResnet
from tbv.models.late_fusion import LateFusionSiameseCEResnet
from tbv.models.no_fusion import SingleModalityCEResnet, SingleModalityLabelmapCEResnet
from tbv.models.siamese_resnet import SiameseTripletResnet, SiameseContrastiveResnet
from tbv.rendering_config import RenderingConfig
from tbv.utils import tbv_transform
from tbv.utils.tbv_transform import unnormalize_img
from tbv.training.mcd_dataset import McdData
from tbv.rendering_config import SensorViewpoint
from tbv.training_config import TrainingConfig


logger = logger_utils.get_logger()


def cross_entropy_forward(
    model: nn.Module,
    args: TrainingConfig,
    split: str,
    x: Tensor,
    xstar: Tensor,
    labelmap: Tensor,
    y: Tensor,
    log_ids: Optional[List[str]] = None,
    timestamps: Optional[Tensor] = None,
    run_gradcam: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Run inputs through model for a cross-entropy-loss based training paradigm.

    Args:
        model: Pytorch model.
        args: training config.
        split:
        x: tensor of shape (N,H,W,3) representing sensor data for each of N examples.
        xstar: tensor of shape (N,H,W,3) representing rendered maps for each of N examples.
        labelmap: tensor of shape (N,H,W) representing a semantic segmentation label map for each of N examples.
        y: tensor of shape (N,) representing a
        log_ids: corresponding vehicle log ID per example.
        timestamps: corresponding nanosecond timestamp per example.
        run_gradcam:

    Returns:
        probs: tensor of shape (N,C) representing 
        loss
    """
    if args.model_name in ["SingleModalityCEResnet", "SingleModalityLabelmapCEResnet"]:
        # feed in map data only or label map data only, not the sensor data.

        if args.fusion_modalities == ["map"]:
            input = xstar
        elif args.fusion_modalities == ["sensor"]:
            input = x
        elif args.fusion_modalities == ["semantics"]:
            input = labelmap
        else:
            raise RuntimeError("Unknown single modality choice")

        if split == "train":
            logits = model(input)
            probs = torch.nn.functional.softmax(logits.clone(), dim=1)
            loss = torch.nn.functional.cross_entropy(logits, y.squeeze())
        else:
            with torch.no_grad():
                logits = model(input)
                probs = torch.nn.functional.softmax(logits.clone(), dim=1)
                loss = torch.nn.functional.cross_entropy(logits, y.squeeze())

            if run_gradcam:
                view_gradcam_results(
                    log_ids,
                    timestamps.numpy(),
                    args,
                    model,
                    x.detach().clone(),
                    xstar.detach().clone(),
                    labelmap.detach().clone(),
                    probs=probs.detach().clone(),
                )

    elif args.model_name in ["LateFusionSiameseCEResnet", "EarlyFusionCEResnet"]:

        if set(list(args.fusion_modalities)) == set(["semantics", "map"]):
            input1 = labelmap
            input2 = xstar

        elif set(list(args.fusion_modalities)) == set(["sensor", "map"]):
            input1 = x
            input2 = xstar

        else:
            raise RuntimeError("Invalid fusion modalities provided via config.")

        if split == "train":
            logits = model(input1, input2)
            probs = torch.nn.functional.softmax(logits.clone(), dim=1)
            loss = torch.nn.functional.cross_entropy(logits, y.squeeze())

        else:
            with torch.no_grad():
                logits = model(input1, input2)
                probs = torch.nn.functional.softmax(logits.clone(), dim=1)
                loss = torch.nn.functional.cross_entropy(logits, y.squeeze())

            if run_gradcam:
                view_gradcam_results(
                    log_ids=log_ids,
                    timestamps=timestamps.numpy(),
                    args=args,
                    model=model,
                    x=x.detach().clone(),
                    xstar=xstar.detach().clone(),
                    labelmap=labelmap.detach().clone(),
                    probs=probs.detach().clone(),
                )

    elif args.model_name in ["EarlyFusionCEResnetWLabelMap"]:
        if split == "train":
            logits = model(x, xstar, labelmap)
            probs = torch.nn.functional.softmax(logits.clone(), dim=1)
            loss = torch.nn.functional.cross_entropy(logits, y.squeeze())

        else:
            with torch.no_grad():
                logits = model(x, xstar, labelmap)
                probs = torch.nn.functional.softmax(logits.clone(), dim=1)
                loss = torch.nn.functional.cross_entropy(logits, y.squeeze())

            if run_gradcam:
                view_gradcam_results(
                    log_ids,
                    timestamps.numpy(),
                    args,
                    model,
                    x.detach().clone(),
                    xstar.detach().clone(),
                    labelmap.detach().clone(),
                    probs=probs.detach().clone(),
                )

    return probs, loss


def view_gradcam_results(
    log_ids: List[str],
    timestamps: np.ndarray,
    args: TrainingConfig,
    model: nn.Module,
    x: Tensor,
    xstar: Tensor,
    labelmap: Tensor,
    probs: Tensor,
) -> None:
    """Save image grids showing inputs and corresponding GradCAM activations.

    Args:
        log_ids: corresponding vehicle log ID per example.
        timestamps: corresponding nanosecond timestamp per example.
        args: training config.
        model: Pytorch model.
        x: tensor of shape (N,H,W,3) representing sensor data for each of N examples.
        xstar: tensor of shape (N,H,W,3) representing rendered maps for each of N examples.
        labelmap: tensor of shape (N,H,W) representing a semantic segmentation label map for each of N examples.
        probs: tensor of shape ()
    """
    N = x.shape[0]
    y_hat = torch.argmax(probs, dim=1).cpu().numpy()
    pred_mismatch_idxs = np.where(y_hat == 0)[0]

    for j in pred_mismatch_idxs:

        log_id = log_ids[j]
        timestamp = timestamps[j]

        mean, std = get_imagenet_mean_std()

        # these will go back into the network, again
        sensor_img = x[j, :, :, :].unsqueeze(0)
        map_img = xstar[j, :, :, :].unsqueeze(0)
        semantic_img = labelmap[j, :, :, :].unsqueeze(0)

        rgb_img = sensor_img.clone().squeeze()
        map_rgb_img = map_img.clone().squeeze()

        unnormalize_img(rgb_img, mean, std)
        unnormalize_img(map_rgb_img, mean, std)
        rgb_img = rgb_img.cpu().numpy()
        rgb_img = np.transpose(rgb_img, (1, 2, 0)).astype(np.uint8)

        map_rgb_img = map_rgb_img.cpu().numpy()
        map_rgb_img = np.transpose(map_rgb_img, (1, 2, 0)).astype(np.uint8)

        if args.model_name == "SingleModalityCEResnet":
            feature_module = model.module.net.resnet.layer4
        else:
            feature_module = model.module.resnet.layer4

        captum_utils.get_gradcam_results(
            log_id,
            timestamp,
            args,
            rgb_img=rgb_img,
            map_rgb_img=map_rgb_img,
            normalized_inputs=[sensor_img, map_img, semantic_img],
            model=model.module,
            feature_module=feature_module,
            target_layer_names=["1"],  # ResNet-18 has just two blocks
            use_cuda=True,
        )


def cross_entropy_forward_two_head(
    model: nn.Module, args: TrainingConfig, split: str, x: Tensor, xstar: Tensor, y: Tensor, gt_is_match: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        model: Pytorch model.
        args: training config.
        split:
        x: tensor of shape (N,H,W,3) representing sensor data for each of N examples.
        xstar: tensor of shape (N,H,W,3) representing rendered maps for each of N examples.
        y:
        gt_is_match:

    Returns:
        is_match_probs
        class_probs
        loss
    """
    if args.model_name != "EarlyFusionTwoHeadResnet":
        raise RuntimeError

    if split == "train":
        is_match_logits, class_logits = model(x, xstar)
        is_match_probs = torch.nn.functional.softmax(is_match_logits.clone(), dim=1)
        class_probs = torch.nn.functional.softmax(class_logits.clone(), dim=1)

        is_match_loss = torch.nn.functional.cross_entropy(is_match_logits, gt_is_match.squeeze())
        class_loss = per_class_sigmoid_loss(class_logits, y)
        loss = is_match_loss + args.aux_loss_weight * class_loss

    else:
        with torch.no_grad():
            is_match_logits, class_logits = model(x, xstar)
            is_match_probs = torch.nn.functional.softmax(is_match_logits.clone(), dim=1)
            class_probs = torch.nn.functional.softmax(class_logits.clone(), dim=1)

            is_match_loss = torch.nn.functional.cross_entropy(is_match_logits, gt_is_match.squeeze())
            class_loss = per_class_sigmoid_loss(class_logits, y)
            loss = is_match_loss + args.aux_loss_weight * class_loss

    return is_match_probs, class_probs, loss


def per_class_sigmoid_loss(
    class_logits: Tensor,
    y_gt: Tensor,
):
    """SigmoidLoss

    Args:
        class_logits: tensor of shape (N x C)
        y_gt: tensor of shape (N,), with values in [0,C-1]

    Returns:
        tensor of shape ()
    """
    n, num_classes = class_logits.shape
    y_onehot = torch.zeros(n, num_classes)
    y_onehot[torch.arange(n), y_gt] = 1
    y_onehot = y_onehot.float().cuda()

    return F.binary_cross_entropy_with_logits(class_logits, y_onehot, reduction="mean")


def sigmoid(x: Tensor):
    """ """
    return 1 / (1 + torch.exp(-x))


def bce(prob: Tensor, y: Tensor):
    """prob and y should be of same shape"""
    return -(y * torch.log(prob) + (1 - y) * torch.log(1 - prob))


def bce_w_logits(x: Tensor, y: Tensor):
    """ """
    N, C = x.shape
    probs = sigmoid(x)
    loss = bce(probs, y)
    return loss.sum() / (N * C)


def bce_w_logits_pytorch(x: Tensor, y: Tensor) -> Tensor:
    """Sigmoid layer and the BCELoss"""
    m = nn.Sigmoid()
    loss_fn = nn.BCELoss()
    loss_val = loss_fn(m(x), y)
    return loss_val.sum()


def print_time_remaining(batch_time: AverageMeter, current_iter: int, max_iter: int) -> None:
    """ """
    remain_iter = max_iter - current_iter
    remain_time = remain_iter * batch_time.avg
    t_m, t_s = divmod(remain_time, 60)
    t_h, t_m = divmod(t_m, 60)
    remain_time = f"{int(t_h):02d}:{int(t_m):02d}:{int(t_s):02d}"
    logger.info(f"\tRemain {remain_time}")


def poly_learning_rate(base_lr: float, curr_iter: int, max_iter: int, power: float = 0.9) -> float:
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def get_train_transform_list(args: TrainingConfig, viewpoint: SensorViewpoint) -> List[Callable]:
    """Get data transforms for training split.

    We treat the semantic BEV label map as an RGB image, and normalize it.

    In the ego-view, we derive binary masks from the semantic segmentation label map, and do not normalize it.
    """
    mean, std = get_imagenet_mean_std()

    transform_list = []
    if viewpoint == SensorViewpoint.EGOVIEW:
        transform_list.extend([tbv_transform.ResizeLabelMapTriplet(), tbv_transform.CropBottomSquareTriplet()])
        if args.blur_input:
            transform_list.extend([tbv_transform.RandomGaussianBlurTriplet()])

        transform_list.extend(
            [
                tbv_transform.ResizeTriplet((args.resize_h, args.resize_w)),
                tbv_transform.RandomHorizontalFlipTriplet(),
                tbv_transform.CropTriplet(
                    [args.train_h, args.train_w], crop_type="rand", padding=mean, grayscale_labelmap=True
                ),
                tbv_transform.ToTensorTriplet(grayscale_labelmap=args.viewpoint == SensorViewpoint.EGOVIEW),
                tbv_transform.BinarizeLabelMapTriplet(),
                tbv_transform.NormalizeTriplet(
                    normalize_labelmap=args.viewpoint == SensorViewpoint.BEV, mean=mean, std=std
                ),
            ]
        )
        # cannot rotate much unless we place a L2-norm constraint, instead of infty-norm

    elif viewpoint == SensorViewpoint.BEV:

        if args.blur_input:
            transform_list.extend([tbv_transform.RandomGaussianBlurTriplet()])

        transform_list.extend([tbv_transform.ResizeTriplet((args.resize_h, args.resize_w))])
        if args.rotate_min != 0 and args.rotate_max != 0:
            transform_list.extend([tbv_transform.RandRotateTriplet([args.rotate_min, args.rotate_max], padding=mean)])
        transform_list.extend(
            [
                tbv_transform.RandomHorizontalFlipTriplet(),
                tbv_transform.RandomVerticalFlipTriplet(),
                tbv_transform.CropTriplet(
                    [args.train_h, args.train_w], crop_type="rand", padding=mean, grayscale_labelmap=False
                ),  # bev has rgb labelmap
                tbv_transform.ToTensorTriplet(grayscale_labelmap=viewpoint == SensorViewpoint.EGOVIEW),
                tbv_transform.NormalizeTriplet(normalize_labelmap=viewpoint == SensorViewpoint.BEV, mean=mean, std=std),
            ]
        )

    # if len(args.modalities_for_random_dropout) > 0:
    #   # every example will have 1/N of the specified modalities missing
    #   transform_list.extend(
    #       [
    #           tbv_transform.RandomModalityDropoutWithoutReplacementTriplet(args.modalities_for_random_dropout, args.modality_dropout_prob)
    #       ])

    transform_list.extend(
        [
            tbv_transform.RandomSemanticsDropoutTriplet(p=args.independent_semantics_dropout_prob),
            tbv_transform.RandomMapDropoutTriplet(p=args.independent_map_dropout_prob),
        ]
    )

    return transform_list


def get_val_test_transform_list(args: TrainingConfig, viewpoint: SensorViewpoint) -> List[Callable]:
    """Get data transforms for synthetic val, real val or test split.

    No horizontal or vertical flips or applied. A center crop is used, instead of a random crop.
    """
    mean, std = get_imagenet_mean_std()

    transform_list = [
        tbv_transform.ResizeTriplet((args.resize_h, args.resize_w)),
        tbv_transform.CropTriplet(
            [args.train_h, args.train_w],
            crop_type="center",
            padding=mean,
            grayscale_labelmap=viewpoint == SensorViewpoint.EGOVIEW,
        ),
        tbv_transform.ToTensorTriplet(grayscale_labelmap=viewpoint == SensorViewpoint.EGOVIEW),
    ]
    if viewpoint == SensorViewpoint.EGOVIEW:
        transform_list.extend([tbv_transform.BinarizeLabelMapTriplet()])

    transform_list.extend(
        [tbv_transform.NormalizeTriplet(normalize_labelmap=viewpoint == SensorViewpoint.BEV, mean=mean, std=std)]
    )
    return transform_list


def get_img_transform_list(args: TrainingConfig, split: str, viewpoint: SensorViewpoint) -> Callable:
    """Return the input data transform for training (w/ data augmentations)

    Note: We do not use any random scaling, since we want to keep the data metric.

    Args:
        args: config for training/testing a model.
        split: dataset split.
        viewpoint: either bird's eye view, or ego-view.

    Return:
        List of transforms
    """
    if viewpoint not in [SensorViewpoint.EGOVIEW, SensorViewpoint.BEV]:
        raise RuntimeError("Unknown viewpoint")

    if split not in ["train", "synthetic_val", "val", "test"]:
        raise RuntimeError("Unknown split. Quitting ...")

    if split == "train":
        transform_list = get_train_transform_list(args, viewpoint=viewpoint)

    elif split in ["synthetic_val", "val", "test"]:
        transform_list = get_val_test_transform_list(args, viewpoint=viewpoint)

    return tbv_transform.ComposeTriplet(transform_list)


def get_optimizer(args: TrainingConfig, model: nn.Module):
    """ """
    # optimizer = torch.optim.SGD([{'params':
    #   filter(lambda p: p.requires_grad,
    #   model.parameters()),
    #   'lr': args.base_lr}],
    #   lr=args.base_lr,
    #   momentum=args.momentum,
    #   weight_decay=args.weight_decay,
    #   # nesterov=config.TRAIN.NESTEROV,
    #   )

    if args.optimizer_algo == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    elif args.optimizer_algo == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    else:
        raise RuntimeError("Unknown optimization algorithm")

    return optimizer


def get_dataloader(
    dataset_args: RenderingConfig,
    training_args: TrainingConfig,
    split: str,
    eval_categories: Optional[List[str]] = None,
    filter_eval_by_visibility: bool = False,
    save_visualizations: bool = False,
) -> torch.utils.data.DataLoader:
    """

    Args:
        dataset_args: config for dataset setup.
        training_args: config for training/testing a model.
        split: dataset split.
        eval_categories: categories to consider when selecting frames to evaluate on.
        filter_eval_by_visibility: whether to evaluate only on visible nearby portions of the scene,
           or to evaluate on *all* nearby portions of the scene.
           (only useful for the "test" split.)
        save visualizations: whether to save side-by-side visualizations of data examples,
            i.e. an image with horizontally stacked (sensor image, map image, blended combination).

    Returns:
        split_loader: data loader for the specified dataset split.
    """
    if split not in ["train", "synthetic_val", "val", "test"]:
        raise RuntimeError("Invalid data split requested.")

    data_transform = get_img_transform_list(training_args, split=split, viewpoint=dataset_args.viewpoint)

    if split in ["val","test"] and not isinstance(eval_categories, list):
        raise ValueError("`eval_categories` must represent a list of categories when testing on the test split.")

    if split in ["train", "synthetic_val"] and eval_categories is not None:
        raise ValueError("`eval_categories` must be set to None when training/evaluating on train or synthetic val splits.")

    split_data = McdData(
        split=split,
        transform=data_transform,
        dataset_args=dataset_args,
        training_args=training_args,
        filter_eval_by_visibility=filter_eval_by_visibility,
        eval_categories=eval_categories,
        loss_type=training_args.loss_type,
        save_visualizations=save_visualizations,
    )

    drop_last = True if split == "train" else False

    # note: we don't shuffle for the "val" or "test" splits
    split_loader = torch.utils.data.DataLoader(
        split_data,
        batch_size=training_args.batch_size_test if split in ["val","test"] else training_args.batch_size,
        shuffle=True if split == "train" else False,
        num_workers=training_args.workers,
        pin_memory=True,
        drop_last=drop_last,
        sampler=None,
    )
    return split_loader


def get_model(args: TrainingConfig, viewpoint: SensorViewpoint) -> nn.Module:
    """Model factory to create a Pytorch model object, without TbV-specific weights.

    Args:
        args: configuration file for training/testing a model.
        viewpoint: dataset rendering perspective.

    Returns:
        model: Pytorch model with random or pretrained weights. We always wrap the model in Pytorch's DataParallel.
    """
    if args.loss_type == "triplet":
        model = SiameseTripletResnet(args.num_layers, args.pretrained)
    elif args.loss_type == "contrastive":
        model = SiameseContrastiveResnet(args.num_layers, args.pretrained)
    elif args.loss_type == "cross_entropy":
        if args.model_name == "LateFusionSiameseCEResnet":
            model = LateFusionSiameseCEResnet(
                args.num_layers, args.pretrained, args.num_ce_classes, late_fusion_operator="concat"
            )
        elif args.model_name == "SingleModalityCEResnet":
            model = SingleModalityCEResnet(args.num_layers, args.pretrained, args.num_ce_classes)
        elif args.model_name == "EarlyFusionCEResnet":
            model = EarlyFusionCEResnet(
                args.num_layers, args.pretrained, args.num_ce_classes, viewpoint, args.fusion_modalities
            )
        elif args.model_name == "EarlyFusionTwoHeadResnet":
            model = EarlyFusionTwoHeadResnet(args.num_layers, args.pretrained, args.num_finegrained_classes)
        elif args.model_name == "EarlyFusionCEResnetWLabelMap":
            model = EarlyFusionCEResnetWLabelMap(args.num_layers, args.pretrained, args.num_ce_classes, viewpoint)
        elif args.model_name == "SingleModalityLabelmapCEResnet":
            model = SingleModalityLabelmapCEResnet(args.num_layers, args.pretrained, args.num_ce_classes)
        else:
            raise RuntimeError("Unknown model type")
    else:
        raise RuntimeError("Unknown loss type")

    logger.info(model)
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        logger.info("CUDA unavailable; inference will be slow.")

    model = torch.nn.DataParallel(model)

    return model


def acc_improved_over_last_k_epochs(arr: List[float], k: int) -> bool:
    """k strikes -- learning rate needs to be decayed soon"""
    return max(arr[:-k]) < max(arr[-k:])


class BinaryClassificationAverageMeter:
    """An AverageMeter designed specifically for evaluating binary classification results."""

    def __init__(self) -> None:
        """Initialize object."""
        self.rec_meter = AverageMeter()
        self.prec_meter = AverageMeter()
        self.f1_meter = AverageMeter()

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update the running mean for each metric."""
        target = target.cpu().numpy()
        pred = pred.cpu().numpy()

        n = target.size

        prec = sklearn.metrics.precision_score(target, pred)
        rec = sklearn.metrics.recall_score(target, pred)
        f1 = sklearn.metrics.f1_score(target, pred)

        self.rec_meter.update(val=rec, n=n)
        self.prec_meter.update(val=prec, n=n)
        self.f1_meter.update(val=f1, n=n)

    def get_metrics(self) -> Tuple[float, float, float]:
        """ """
        return self.prec_meter.avg, self.rec_meter.avg, self.f1_meter.avg


# sklearn.metrics.precision_recall_curve(y_true, probas_pred, *, pos_label=None, sample_weight=None)
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html

