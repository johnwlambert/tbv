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

from dataclasses import dataclass
from typing import Optional, List

import hydra
from hydra.utils import instantiate


VALID_MODEL_NAMES = [
    "SingleModalityCEResnet",
    "EarlyFusionCEResnet",
    "EarlyFusionCEResnetWLabelMap",
    "LateFusionSiameseCEResnet",
    "EarlyFusionTwoHeadResnet",
]


@dataclass
class TrainingConfig:
    """
    TODO: Could use a hierarchy of configs to place rendering config inside the training config, as member.

    Args:
        model_name:
            EarlyFusionCEResnetWLabelMap
            LateFusionSiameseCEResnet
            SingleModalityCEResnet
            EarlyFusionTwoHeadResnet
            EarlyFusionCEResnet

        fusion_modalities: modalities to use as model input. Choices include: "sensor", "semantics", "map"

        workers: number of CPU workers for data-loading during training (using train.py).
        batch_size: batch size during training.

        train_h: height of input crop to network, in pixels.
        train_w: width of input crop to network, in pixels.
        rotate_min: angle – Rotation angle in degrees. Positive values mean counter-clockwise
            rotation (the coordinate origin is assumed to be the top-left corner).
        rotate_max:

        lr_annealing_strategy: strategy for annealing learning rate, either `reduce_on_plateau` vs. `poly`
        reduce_on_plateau_power: if using `reduce_on_plateau`, amount to decay LR by when plateau is achieved.
    """

    # model parameters
    model_name: str
    arch: str
    pretrained: bool
    num_layers: int
    model_save_dirpath: str
    num_fc_layers: int

    # inputs
    fusion_modalities: List[str]
    independent_semantics_dropout_prob: float
    independent_map_dropout_prob: float
    use_multiple_negatives_per_sensor_img: bool

    # resource-specific hyperparameters
    workers: int
    batch_size: int
    batch_size_test: int

    # data aug
    train_h: int
    train_w: int
    resize_h: int
    resize_w: int
    blur_input: bool
    rotate_min: float
    rotate_max: float

    # optimization
    optimizer_algo: str
    base_lr: float
    poly_lr_power: float
    num_epochs: int
    resume_iter: int
    weight_decay: float
    momentum: float
    loss_type: str
    lr_annealing_strategy: str

    # TbV taxonomy
    num_ce_classes: int
    num_finegrained_classes: int
    aux_loss_weight: Optional[float] = None  # only for 2-head model

    # not used for published experiments, only for metric learning.
    contrastive_tp_dist_thresh: Optional[float] = None
    reduce_on_plateau_power: Optional[float] = 0.1

    def __post_init__(self) -> None:
        """Verify certain fields."""
        valid_fusion_modalities = set(["sensor", "semantics", "map"])
        if not set(self.fusion_modalities).issubset(valid_fusion_modalities):
            raise ValueError(f"Invalid input modalities: {self.fusion_modalities}")

        if len(self.fusion_modalities) == 1:
            assert self.model_name == "SingleModalityCEResnet"

        if self.model_name == "SingleModalityCEResnet":
            assert len(self.fusion_modalities) == 1

        elif self.model_name == "EarlyFusionCEResnetWLabelMap":
            assert len(self.fusion_modalities) == 3
            assert set(self.fusion_modalities) == set(["sensor", "semantics", "map"])

        assert self.model_name in VALID_MODEL_NAMES

        assert isinstance(self.num_layers, int)
        assert isinstance(self.workers, int)
        assert isinstance(self.batch_size, int)


def load_training_config(config_name: str) -> TrainingConfig:
    """Get experiment config for training a map change detection model, in either BEV or ego-view."""
    with hydra.initialize_config_module(config_module="tbv.training_configs"):
        # config is relative to the tbv module
        cfg = hydra.compose(config_name=config_name)
        config: TrainingConfig = instantiate(cfg.TrainingConfig)

    return config
