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
TODO: properly document the unit tests.
"""

import numpy as np

import tbv.utils.triangle_grid_utils as triangle_grid_utils


def test_prune_triangles_zero_yaw() -> None:
    """
    Effective ray directions are
        left_ray_dir = np.array([1,1,0])
        right_ray_dir = np.array([1,-1,0])
    giving a 90 degree f.o.v. to the frustum.
    """
    fov_theta = np.pi / 2  # 90 degrees

    range_m = 1
    tris = triangle_grid_utils.get_flat_plane_grid_triangles(range_m)

    # first centroid must be array([[-0.66666667, -0.66666667],

    yaw = 0  # degrees or radians

    # some small floating point error is introduced in the rotation matrix specification
    frustum_triangles, inside_frustum = triangle_grid_utils.prune_triangles_to_2d_frustum(tris, yaw, fov_theta, margin=1e-10)

    gt_inside_frustum = np.array([False, False, False, False, False, True, True, True], dtype=bool)
    assert np.allclose(inside_frustum, gt_inside_frustum)
    assert len(frustum_triangles) == 3

    # fmt: off
    gt_frustum_triangles = [
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, -1.0, 0.0]),
            np.array([1.0, 0.0, 0.0])
        ],
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0])
        ],
        [
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0])
        ],
    ]
    # fmt: on
    for i in range(3):  # triangles
        for v in range(3):  # vertices
            assert np.allclose(gt_frustum_triangles[i][v], frustum_triangles[i][v])


def test_prune_triangles_back_frustum():
    """
    Look backwards (rotated around by 180 degrees)
    """
    fov_theta = np.pi / 2  # 90 degrees

    range_m = 1
    tris = triangle_grid_utils.get_flat_plane_grid_triangles(range_m)

    yaw = np.pi  # radians

    # some small floating point error is introduced in the rotation matrix specification
    frustum_triangles, inside_frustum = triangle_grid_utils.prune_triangles_to_2d_frustum(tris, yaw, fov_theta, margin=1e-10)

    gt_inside_frustum = np.array([True, True, True, False, False, False, False, False], dtype=bool)
    assert np.allclose(inside_frustum, gt_inside_frustum)
    assert len(frustum_triangles) == 3
