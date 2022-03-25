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
Unit tests to ensure that configs (for rendering training/test data) can be loaded properly.
"""

import omegaconf
import pytest

import tbv.rendering_config as rendering_config
from tbv.rendering_config import EgoviewRenderingConfig, BevRenderingConfig


def test_load_rendering_config_bev_file() -> None:
    """Config should be correctly loaded as BevRenderingConfig."""
    config_name = "render_2021_09_04_bev_synthetic_config_t5820.yaml"

    config = rendering_config.load_rendering_config(config_name)
    assert isinstance(config, BevRenderingConfig)


def test_load_rendering_config_egoview() -> None:
    """Config should be correctly loaded as EgoviewRenderingConfig."""
    config_name = "render_2021_09_03_egoview_synthetic_config_v1.yaml"

    config = rendering_config.load_rendering_config(config_name)
    assert isinstance(config, EgoviewRenderingConfig)


def test_load_egoview_rendering_config_w_bev_file() -> None:
    """Config should NOT be correctly loaded (cannot load BEV config w/ egoview loader)."""
    config_name = "render_2021_09_04_bev_synthetic_config_t5820.yaml"

    with pytest.raises(omegaconf.errors.ConfigAttributeError):
        config = rendering_config.load_egoview_rendering_config(config_name)
    # assert isinstance(config, BevRenderingConfig)
