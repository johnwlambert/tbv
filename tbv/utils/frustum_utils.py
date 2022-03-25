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
