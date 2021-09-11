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

from typing import Tuple

import numpy as np

from argoverse.utils.calibration import CameraConfig
from argoverse.utils.se3 import SE3


def get_frustum_parameters(camera_config: CameraConfig, verbose: bool = False) -> Tuple[float, float]:
    """Compute the field of view of a camera frustum to use for view frustum culling during rendering.

    R takes the x axis to be a vector equivalent to the first column
    of R. Similarly, the y and z axes are transformed to be the second
    and third columns.

    Args:
        camera_config:
        verbose: whether to dump to stdout information about frustum field of view.

    Returns:
        cam_yaw_ego: float representing clockwise angle from x=0 (in radians)
        fov_theta: angular extent of camera's field of view (measured in radians)
    """
    camera_SE3_egovehicle = camera_config.extrinsic
    camera_R_egovehicle = camera_SE3_egovehicle[:3, :3]
    camera_SE3_egovehicle = SE3(rotation=camera_R_egovehicle, translation=np.zeros(3))

    I = np.eye(3)
    transformed_axes = camera_SE3_egovehicle.transform_point_cloud(I)

    new_x_axis = transformed_axes[:, 2]
    dx, dy, dz = new_x_axis

    cam_yaw_ego = np.arctan2(dy, dx)

    fx = camera_config.intrinsic[0, 0]
    fy = camera_config.intrinsic[1, 1]

    # return in radians
    fov_theta = 2 * np.arctan(camera_config.img_width / fx)

    if verbose:
        logger.info(f"\tComputed yaw={np.rad2deg(cam_yaw_ego):.2f} for this ring camera.")
        logger.info(f"\tComputed fov={np.rad2deg(fov_theta):.2f} for this ring camera.")

    return cam_yaw_ego, fov_theta


def get_frustum_side_normals(fov_theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Get normal vectors for left and right clipping planes of a camera/view frustum.

                   left clipping plane
                  /
                 X
                /  \\
              /     N l_normal
            /
    origin O ------ +x
            \
              \\   N r_normal
               \\ //
                 X
                  \\
                    right clipping plane

    Args:
        fov_theta: field of view angle for camera/view frustum., must be in radians.

    Returns:
        l_normal: normal to left clipping plane. normal points into frustum, unit-length vectors
        r_normal: normal to right clipping plane.
    """
    # find point on unit circle -- ray to this point from the origin corresponds to left clipping plane direction
    left_ray_dir = np.cos(fov_theta / 2), np.sin(fov_theta / 2)
    x1, y1 = left_ray_dir[:2]
    l_normal = np.array([y1, -x1])  # rotate by 90 deg to get normal pointing into frustum.

    # find point on unit circle -- ray to this point from the origin corresponds to right clipping plane direction
    right_ray_dir = np.cos(-fov_theta / 2), np.sin(-fov_theta / 2)
    x2, y2 = right_ray_dir[:2]
    r_normal = np.array([-y2, x2])
    return l_normal, r_normal
