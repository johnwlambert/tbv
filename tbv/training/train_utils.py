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

# from mseg_semantic.utils.training_utils import poly_learning_rate

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
        model:
        args:
        split:
        x: tensor of shape ()
        xstar: tensor of shape ()
        labelmap: tensor of shape ()
        y: tensor of shape ()
        log_ids:
        timestamps:
        run_gradcam:

    Returns:
        probs
        loss
    """
    if args.model_name in ["SingleModalityCEResnet", "SingleModalityLabelmapCEResnet"]:
        # feed in map data only, not the sensor data

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
                    log_ids,
                    timestamps.numpy(),
                    args,
                    model,
                    x.detach().clone(),
                    xstar.detach().clone(),
                    labelmap.detach().clone(),
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
    args,
    model,
    x: Tensor,
    xstar: Tensor,
    labelmap: Tensor,
    probs: Tensor,
) -> None:
    """Save image grids showing inputs and corresponding GradCAM activations.

    Args:
        log_ids:
        timestamps:
        args
        model:
        x: tensor of shape ()
        xstar: tensor of shape ()
        labelmap: tensor of shape ()
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
    model: nn.Module, args, split: str, x: Tensor, xstar: Tensor, y: Tensor, gt_is_match: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        model:
        args:
        split:
        x:
        xstar:
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


def test_per_class_sigmoid_loss() -> None:
    """ """
    n = 5
    num_classes = 6

    class_logits = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 3
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 4
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 5
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 0
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 1
        ]
    ).reshape(n, num_classes)
    class_logits = torch.from_numpy(class_logits).float()

    y = np.array([[3], [4], [5], [0], [1]]).reshape(n)
    y = torch.from_numpy(y)

    # class_logits = torch.tensor(
    #   [
    #       [0,1],
    #       [1,0],
    #       [.5,.5]
    #   ])
    # y_onehot = torch.tensor(
    #   [
    #       [0,1],
    #       [1,0],
    #       [.5,.5]
    #   ])
    # Expected pytorch loss: tensor(0.5768)

    loss = per_class_sigmoid_loss(class_logits, y)
    print("Pytorch loss: ", loss)

    # expected_loss = bce_w_logits(class_logits,y_onehot)
    # print('Expected loss:', expected_loss)

    # expected_pytorch_loss = bce_w_logits_pytorch(class_logits, y_onehot)
    # print('Expected pytorch loss:', expected_pytorch_loss)

    assert np.isclose(loss.item(), 0.6298, atol=1e-4)
    # assert np.isclose(expected_loss.item(), 0.6298, atol=1e-4)
    # assert np.isclose(expected_pytorch_loss.item(), 0.6298, atol=1e-4)


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


def test_poly_learning_rate() -> None:
    """ """
    lr = poly_learning_rate(0.001, curr_iter=0, max_iter=100, power=0.9)
    assert np.isclose(lr, 0.001)

    lr = poly_learning_rate(0.001, curr_iter=10, max_iter=100, power=0.9)
    assert np.isclose(lr, 0.0009095325760829623)

    lr = poly_learning_rate(0.001, curr_iter=99, max_iter=100, power=0.9)
    assert np.isclose(lr, 1.5848931924611145e-05)

    lr = poly_learning_rate(0.001, curr_iter=100, max_iter=100, power=0.9)
    assert np.isclose(lr, 0.0)


def get_train_transform_list(args: TrainingConfig, viewpoint: str) -> List[Callable]:
    """Get data transforms for training split"""
    mean, std = get_imagenet_mean_std()

    transform_list = []
    if args.viewpoint == SensorViewpoint.EGOVIEW:
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
                tbv_transform.NormalizeTriplet(normalize_labelmap=args.viewpoint == SensorViewpoint.BEV, mean=mean, std=std),
            ]
        )
        # cannot rotate much unless we place a L2-norm constraint, instead of infty-norm

    elif args.viewpoint == SensorViewpoint.BEV:

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


def get_val_test_transform_list(args: TrainingConfig, viewpoint: str) -> List[Callable]:
    """Get data transforms for val or test split"""
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

    transform_list.extend([tbv_transform.NormalizeTriplet(normalize_labelmap=viewpoint == SensorViewpoint.BEV, mean=mean, std=std)])
    return transform_list


def get_img_transform_list(args: TrainingConfig, split: str, viewpoint: str) -> Callable:
    """Return the input data transform for training (w/ data augmentations)

    Note: We do not use any random scaling, since we want to keep the data metric.

    Args:
        args: config for training/testing a model.
        split: dataset split.
        viewpoint: either "bev" or "egoview"

    Return:
        List of transforms
    """
    if viewpoint not in [SensorViewpoint.EGOVIEW, SensorViewpoint.BEV]:
        raise RuntimeError("Unknown viewpoint")

    if split not in ["train", "val", "test"]:
        raise RuntimeError("Unknown split. Quitting ...")

    if split == "train":
        transform_list = get_train_transform_list(args, viewpoint)

    elif split in ["val", "test"]:
        transform_list = get_val_test_transform_list(args, viewpoint)

    return tbv_transform.ComposeTriplet(transform_list)


def get_optimizer(args, model):
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
    if split not in ["train", "val", "test"]:
        raise RuntimeError("Invalid data split requested.")

    data_transform = get_img_transform_list(training_args, split=split, viewpoint=dataset_args.viewpoint)

    if split == "test" and not isinstance(eval_categories, list):
        raise ValueError("`eval_categories` must represent a list of categories when testing on the test split.")

    if split in ["train", "val"] and eval_categories is not None:
        raise ValueError("`eval_categories` must be set to None when training/evaluating on train/val splits.")

    # TODO: pass the train vs. val list for log_ids in each section
    split_data = McdData(
        split=split,
        transform=data_transform,
        args=dataset_args,
        filter_eval_by_visibility=filter_eval_by_visibility,
        eval_categories=eval_categories,
        loss_type=training_args.loss_type,
        save_visualizations=save_visualizations
    )

    drop_last = True if split == "train" else False

    # note: we don't shuffle for the "val" or "test" splits
    split_loader = torch.utils.data.DataLoader(
        split_data,
        batch_size=training_args.batch_size_test if split == "test" else training_args.batch_size,
        shuffle=True if split == "train" else False,
        num_workers=training_args.workers,
        pin_memory=True,
        drop_last=drop_last,
        sampler=None,
    )
    return split_loader


def get_model(args: TrainingConfig, viewpoint: str) -> nn.Module:
    """Model factory to create a Pytorch model object, without TbV-specific weights.

    Args:
        args: configuration file for training/testing a model.
        viewpoint:

    Returns:
        model: Pytorch model with random or pretrained weights.

    We always wrap the model in Pytorch's DataParallel.
    """
    if args.loss_type == "triplet":
        model = SiameseTripletResnet(args.num_layers, args.pretrained)
    elif args.loss_type == "contrastive":
        model = SiameseContrastiveResnet(args.num_layers, args.pretrained)
    elif args.loss_type == "cross_entropy":
        if args.model_name == "LateFusionSiameseCEResnet":
            model = LateFusionSiameseCEResnet(
                args.num_layers, args.pretrained, args.num_ce_classes, args.late_fusion_operator
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


def test_acc_improved_over_last_k_epochs() -> None:
    """ """
    acc = [0, 1, 2, 3, 4, 5, 6]
    has_improved = acc_improved_over_last_k_epochs(acc, k=5)
    assert has_improved

    acc = [2, 2, 2, 2, 2, 2]
    has_improved = acc_improved_over_last_k_epochs(acc, k=5)
    assert not has_improved

    acc = [2, 1, 1, 1, 1, 1]
    has_improved = acc_improved_over_last_k_epochs(acc, k=5)
    assert not has_improved

    acc = [50, 1, 2, 3, 4, 5]
    has_improved = acc_improved_over_last_k_epochs(acc, k=5)
    assert not has_improved

    acc = [50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51]
    has_improved = acc_improved_over_last_k_epochs(acc, k=5)
    assert has_improved


class BinaryClassificationAverageMeter:
    """
    An AverageMeter designed specifically for evaluating binary classification results.
    """

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


def test_classification_average_meter() -> None:
    """The positive class is any map-change category"""

    y_pred = torch.tensor(
        [
            0,  # correct (TN)
            0,  # should have predicted change (mismatch) FN
            1,  # correct (TP)
            1,  # should have predicted no change (FP)
        ]
    )

    y_true = torch.tensor([0, 1, 1, 0])
    cam = BinaryClassificationAverageMeter()
    cam.update(y_pred, y_true)
    prec, rec, f1 = cam.get_metrics()

    tp = 1
    fp = 1
    fn = 1

    gt_prec = tp / (tp + fp)  # divide by # predicted positives
    gt_rec = tp / (tp + fn)  # divide by actual # positives
    gt_f1 = 2 * (gt_prec * gt_rec) / (gt_prec + gt_rec)

    assert prec == gt_prec
    assert rec == gt_rec
    assert f1 == gt_f1


def compute_mean_accuracy(
    probs: Tensor, y_true: Tensor, n_class: int, verbose: bool
) -> Tuple[float, np.ndarray]:
    """over all classes

    Args:
        probs: (N,C)
        y_true: (N,)

    Returns:
        mAcc
        accs (C,)
    """
    y_pred = torch.argmax(probs, dim=1)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    correct = y_true == y_pred

    gt_class_freqs, bin_edges = np.histogram(y_true, bins=n_class)
    gt_class_freqs = gt_class_freqs.astype(np.float32) / y_true.size

    if verbose:
        print("\tGT Class frequencies:", [f"{100*freq:.2f}%" for freq in gt_class_freqs])

    accs = np.zeros(n_class)  # per class accuracy
    for c in np.unique(y_true):

        c_idxs = y_true == c
        accs[c] = correct[c_idxs].sum() / c_idxs.sum()

    return accs.mean(), accs


def test_compute_mean_accuracy() -> None:
    """ """

    probs = torch.tensor(
        [
            [0.8, 0.1, 0.1],  # 0
            [0.8, 0.1, 0.1],  # 0
            [0.8, 0.1, 0.1],  # 0
            [0.8, 0.1, 0.1],  # 0
            [0.1, 0.8, 0.1],  # 1
            [0.1, 0.1, 0.8],  # 2
        ]
    )
    y_true = torch.tensor([0, 0, 1, 1, 2, 2])
    n_class = 3

    mAcc, accs = compute_mean_accuracy(probs, y_true, n_class, verbose=True)
    assert mAcc == 0.5

    y_hat = torch.argmax(probs, dim=1)
    sam = SegmentationAverageMeter()
    sam.update_metrics_cpu(pred=y_hat.cpu().numpy(), target=y_true.cpu().numpy(), num_classes=3)
    _, accuracy_class, _, mAcc, _ = sam.get_metrics()
    assert np.isclose(mAcc, 0.5)
    assert np.allclose(accuracy_class, np.array([1, 0, 0.5]))


if __name__ == "__main__":
    # test_per_class_sigmoid_loss()
    # test_compute_mean_accuracy()

    # test_classification_average_meter()

    # test_acc_improved_over_last_k_epochs()
    pass
