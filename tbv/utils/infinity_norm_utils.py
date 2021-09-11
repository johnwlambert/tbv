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

import logging
import numpy as np

import argoverse.utils.interpolate as interp_utils

WPT_INFTY_NORM_INTERP_NUM = 50


def has_pts_in_infty_norm_radius(pts: np.ndarray, window_center: np.ndarray, window_sz: float) -> bool:
    """Check if a map entity has points within a search radius from a single query point.

    Note: This does NOT measure distance by Manhattan distance -- this is the infinity norm.

    Args:
        pts: Nx2 array, representing map entity
        window_center: 1x2, or (2,) array, representing query point
        window_sz: search radius

    Returns:
        boolean array of shape (N,) representing which of the N points lie within the search radius.
    """
    if pts.ndim != 2:
        raise ValueError("`pts` must consist of a sequence of 2d or 3d waypoints, i.e. w/ ndim=2.")

    if pts.shape[1] == 3:
        # take only x,y dimensions
        pts = pts[:, :2]
    assert pts.size % 2 == 0
    assert pts.shape[1] == 2

    if window_center.ndim == 2:
        window_center = window_center.squeeze()
    assert window_center.ndim == 1
    if window_center.size == 3:
        window_center = window_center[:2]
    assert window_center.size == 2
    # reshape just in case was given column vector
    window_center = window_center.reshape(1, 2)

    dists = np.linalg.norm(pts - window_center, ord=np.inf, axis=1)
    return dists.min() < window_sz


def lane_bounds_in_infty_norm_radius(
    right_ln_bnd: np.ndarray, left_ln_bnd: np.ndarray, ego_center: np.ndarray, window_sz: float
) -> bool:
    """ """
    try:
        right_ln_bnd_interp = interp_utils.interp_arc(
            t=WPT_INFTY_NORM_INTERP_NUM, px=right_ln_bnd[:, 0], py=right_ln_bnd[:, 1]
        )
        left_ln_bnd_interp = interp_utils.interp_arc(
            t=WPT_INFTY_NORM_INTERP_NUM, px=left_ln_bnd[:, 0], py=left_ln_bnd[:, 1]
        )
    except Exception as e:
        print("Interpolation attempt failed!")
        logging.exception("Interpolation failed")
        # 1-point line segments will cause trouble later
        right_ln_bnd_interp = right_ln_bnd[:, :2]
        left_ln_bnd_interp = left_ln_bnd[:, :2]

    return has_pts_in_infty_norm_radius(right_ln_bnd_interp, ego_center, window_sz) or has_pts_in_infty_norm_radius(
        left_ln_bnd_interp, ego_center, window_sz
    )
