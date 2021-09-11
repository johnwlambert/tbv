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
Unit tests to verify ray generation utilities.
"""

import numpy as np
import pytest

import tbv.utils.frustum_ray_generation_utils as frustum_ray_gen_utils


def test_compute_pixel_ray_directions_vectorized_invalid_focal_lengths():
    """If focal lengths in the x and y directions do not match, we throw an exception.

    Tests vectorized variant (multiple ray directions.)
    """
    uv = np.array([[12, 2], [12, 2], [12, 2], [12, 2]])
    fx = 10
    fy = 11

    img_w = 20
    img_h = 10
    with pytest.raises(ValueError):
        ray_dirs = frustum_ray_gen_utils.compute_pixel_ray_directions_vectorized(uv, fx, fy, img_w, img_h)


def test_compute_pixel_ray_direction_invalid_focal_lengths():
    """If focal lengths in the x and y directions do not match, we throw an exception.

    Tests non-vectorized variant (single ray direction).
    """
    u = 12
    v = 2
    fx = 10
    fy = 11

    img_w = 20
    img_h = 10
    with pytest.raises(ValueError):
        ray_dirs = frustum_ray_gen_utils.compute_pixel_ray_direction(u, v, fx, fy, img_w, img_h)


def test_compute_pixel_ray_directions_vectorized() -> None:
    """
    Ensure that the ray direction (in camera coordinate frame) for each pixel is computed correctly.

    Small scale test, for just four selected positions in a 10 x 20 px image in (height, width).
    """
    fx = 10
    fy = 10

    # dummy 2d coordinates in the image plane.
    uv = np.array([[12, 2], [12, 2], [12, 2], [12, 2]])

    # principal point is at (10,5)
    img_w = 20
    img_h = 10

    ray_dirs = frustum_ray_gen_utils.compute_pixel_ray_directions_vectorized(uv, fx, fy, img_w, img_h)

    gt_ray_dir = np.array([2, -3, 10.0])
    gt_ray_dir /= np.linalg.norm(gt_ray_dir)

    for i in range(4):
        assert np.allclose(gt_ray_dir, ray_dirs[i])


def test_compute_pixel_ray_directions_vectorized_entireimage() -> None:
    """
    Ensure that the ray direction for each pixel (in camera coordinate frame) is computed correctly.

    Compare all computed rays against non-vectorized variant, for correctness.
    Larger scale test, for every pixel in a 50 x 100 px image in (height, width).
    """
    fx = 10
    fy = 10

    img_w = 100
    img_h = 50

    uv = []
    for u in range(img_w):
        for v in range(img_h):
            uv += [(u, v)]

    uv = np.array(uv)
    assert uv.shape == (img_w * img_h, 2)

    ray_dirs = frustum_ray_gen_utils.compute_pixel_ray_directions_vectorized(uv, fx, fy, img_w, img_h)

    # compare w/ vectorized, should be identical
    for i, ray_dir_vec in enumerate(ray_dirs):
        u, v = uv[i]
        ray_dir_nonvec = frustum_ray_gen_utils.compute_pixel_ray_direction(u, v, fx, fy, img_w, img_h)
        assert np.allclose(ray_dir_vec, ray_dir_nonvec)


def test_compute_pixel_rays() -> None:
    """Ensure that the ray direction (in camera coordinate frame) for a single pixel is computed correctly.

    Small scale test, for just one selected position in a 10 x 20 px image in (height, width).
    For row = 2, column = 12.
    """
    u = 12
    v = 2
    img_w = 20
    img_h = 10
    fx = 10
    fy = 10

    ray_dir = frustum_ray_gen_utils.compute_pixel_ray_direction(u, v, fx, fy, img_w, img_h)

    gt_ray_dir = np.array([2., -3., 10.])
    gt_ray_dir /= np.linalg.norm(gt_ray_dir)

    assert np.allclose(gt_ray_dir, ray_dir)

