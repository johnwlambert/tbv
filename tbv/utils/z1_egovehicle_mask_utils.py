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

import imageio
import matplotlib.pyplot as plt
import numpy as np
from mseg.utils.mask_utils import get_mask_from_polygon


# front-center camera has `portrait-mode` aspect ratio.
TBV_RING_FRONT_CENTER_IMG_HEIGHT = 2048
TBV_RING_FRONT_CENTER_IMG_WIDTH = 1550

TBV_RING_REAR_RIGHT_IMG_HEIGHT = 1550
TBV_RING_REAR_RIGHT_IMG_WIDTH = 2048

TBV_RING_REAR_LEFT_IMG_HEIGHT = 1550
TBV_RING_REAR_LEFT_IMG_WIDTH = 2048


def filter_out_egovehicle(uv: np.ndarray, camera_name: str) -> np.ndarray:
    """Mask out the immediate foreground (pixels belonging to egovehicle) for any camera.

    Note: only 3 of the 7 ring cameras see the egovehicle in their field of view.
    Do not shoot ray into egovehicle hood or body (mask out foreground)

    Args:
        uv: array of shape (N,2)
        camera_name: string representing the name of a ring camera.

    Returns:
        logicals: array of shape (N,) corresponding to pixels NOT capturing the egovehicle.
    """
    # update assumptions
    if camera_name == "ring_front_center":
        egovehicle_mask = get_z1_ring_front_center_mask()
    elif camera_name == "ring_rear_right":
        egovehicle_mask = get_z1_ring_rear_right_mask()
    elif camera_name == "ring_rear_left":
        egovehicle_mask = get_z1_ring_rear_left_mask()

    valid_mask = ~egovehicle_mask

    y = uv[:, 1]
    x = uv[:, 0]
    logicals = valid_mask[y, x] != 0
    return logicals


def get_z1_ring_front_center_mask() -> np.ndarray:
    """Provide mask for the immediate foreground (pixels belonging to egovehicle), for the ring front center camera.

    Returns:
        mask: boolean array of shape (H,W)
    """
    polygon_verts = np.array(
        [[0, 2048], [0, 1887], [303, 1804], [602, 1765], [951, 1780], [1256, 1800], [1544, 1880], [1549, 2048]]
    )
    mask = get_mask_from_polygon(
        polygon_verts, img_h=TBV_RING_FRONT_CENTER_IMG_HEIGHT, img_w=TBV_RING_FRONT_CENTER_IMG_WIDTH
    )
    return mask.astype(bool)


def get_z1_ring_rear_right_mask() -> np.ndarray:
    """Provide mask for the immediate foreground (pixels belonging to egovehicle), for the ring rear right camera.

    Returns:
        mask: boolean array of shape (H,W)
    """
    polygon_verts = np.array(
        [[511, 1549], [511, 1540], [985, 1376], [1203, 1334], [1405, 1305], [2046, 1318], [2047, 1549]]
    )
    mask = get_mask_from_polygon(
        polygon_verts, img_h=TBV_RING_REAR_RIGHT_IMG_HEIGHT, img_w=TBV_RING_REAR_RIGHT_IMG_WIDTH
    )
    return mask.astype(bool)


def get_z1_ring_rear_left_mask() -> np.ndarray:
    """Provide mask for the immediate foreground (pixels belonging to egovehicle), for the ring rear left camera.

    Returns:
        mask: boolean array of shape (H,W)
    """
    polygon_verts = np.array(
        [
            [0, 1359],
            [12, 1359],
            [335, 1330],
            [620, 1322],
            [708, 1355],
            [889, 1364],
            [1036, 1376],
            [1161, 1397],
            [1539, 1540],
            [1539, 1549],
            [0, 1549],
        ]
    )
    mask = get_mask_from_polygon(
        polygon_verts, img_h=TBV_RING_REAR_LEFT_IMG_HEIGHT, img_w=TBV_RING_REAR_LEFT_IMG_WIDTH
    )
    return mask.astype(bool)
