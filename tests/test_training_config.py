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

"""
Unit tests to ensure that configs (for training/testing models) can be loaded properly.
"""

import tbv.training_config as training_config
from tbv.training_config import TrainingConfig


def test_load_training_config_bev_file() -> None:
    """Config should be correctly loaded as TrainingConfig."""
    config_name = "2021_09_09_train_bev.yaml"

    config = training_config.load_training_config(config_name)
    assert isinstance(config, TrainingConfig)
