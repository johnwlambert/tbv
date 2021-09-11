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

from types import SimpleNamespace
from typing import Any, Dict, NamedTuple

import numpy as np
from argoverse.map_representation.map_api_v2 import ArgoverseStaticMapV2
from argoverse.utils.calibration import CameraConfig
from argoverse.utils.se3 import SE3


class LogEgoviewRenderingMetadata(NamedTuple):
    """Helpful info for projection of entities in the ego-view.

    Args:
        depth_map:
        egovehicle_SE3_city:
        log_calib_data:
        camera_name:
        camera_config:
        avm: map, with vector and raster elements, for querying ground height at arbitrary locations.
    """

    depth_map: np.ndarray
    egovehicle_SE3_city: SE3
    city_SE3_egovehicle: SE3
    log_calib_data: Dict[str, Any]
    camera_name: str
    camera_config: CameraConfig
    avm: ArgoverseStaticMapV2


def within_img_bnds(uv: np.ndarray, camera_config: CameraConfig) -> np.ndarray:
    """

    Args:
        uv: array of shape () representing ...
        camera_config

    Returns:
        valid_pts_bool: array of shape () representing ...
    """
    return np.logical_and.reduce(
        [
            0 <= uv[:, 0],  # valid x
            uv[:, 0] < camera_config.img_width,  # valid x
            0 <= uv[:, 1],  # valid y
            uv[:, 1] < camera_config.img_height,  # valid y
        ]
    )


def test_within_img_bnds() -> None:
    """ """
    camera_config_dict = {"img_width": 20, "img_height": 10}
    uv = np.array([[21, 11], [21, 9], [19, 9], [19, 11], [-1, 0], [0, 0]])  # no  # no  # yes  # no  # no  # yes
    camera_config = SimpleNamespace(**camera_config_dict)
    is_valid = within_img_bnds(uv, camera_config)
    gt_is_valid = np.array([False, False, True, False, False, True])
    assert np.allclose(is_valid, gt_is_valid)

