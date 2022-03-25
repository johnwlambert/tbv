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

import numpy as np

import tbv.utils.triangle_grid_utils as triangle_grid_utils
from tbv.rendering.ray_triangle_intersection import ray_triangle_intersect

def test_get_point_rgb_correspondences_raytracing() -> None:
    """Testing only a portion of the function here"""
    # Simulate ring front center
    origin = np.array([1.63517491, 1.27212439e-03, 1.42744074])
    img_h = 2048
    img_w = 1550
    fx = 1683.6659098631715
    fy = 1683.6659098631715

    u = img_w // 2
    v = img_h - 1

    ray_dir = compute_pixel_ray_direction(u, v, fx, fy, img_w, img_h)
    # ray dir is array([ 0.85461312, -0.        , -0.51926526])

    #              x, y, z
    v0 = np.array([1, 10, 0]).astype(np.float32)
    v1 = np.array([1, -10, 0]).astype(np.float32)
    v2 = np.array([100, 0, 0]).astype(np.float32)

    inter_exists, P = ray_triangle_intersect(origin, ray_dir, v0, v1, v2)
    assert inter_exists
    assert np.allclose(
        P,
        np.array(
            [
                1.42 / 0.519 * 0.85 + 1.63,
                0,
                0,
            ]
        ),
        atol=1e-1,
    )

    nearby_triangles = triangle_grid_utils.get_flat_plane_grid_triangles(range_m=5)

    for i, tri in enumerate(nearby_triangles):
        if i % 100 == 0:
            print(f"On {i}/{len(nearby_triangles)}")
        v0, v1, v2 = tri
        inter_exists, P = ray_triangle_intersect(origin, ray_dir, v0, v1, v2)
        if inter_exists:
            assert np.allclose(P, np.array([3.98, 0.0, 0.0]), atol=1e-2)
