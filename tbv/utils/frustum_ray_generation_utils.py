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
Utilities for computing ray directions for all pixels in an image, under perspective camera model.
Rays are provided in the camera coordinate frame (NOT in the egovehicle frame).

Instead of using the principal point from calibration, we use dead center of image plane. This
hack is less accurate but simple.
"""

import numpy as np


def compute_pixel_ray_direction(u: float, v: float, fx: float, fy: float, img_w: int, img_h: int) -> np.ndarray:
    """Generate rays in the camera coordinate frame.

       Find point P on image plane.

                      (x,y,z)-----(x,y,z)
                         \\          |
    outside of frustum <- \\         |
     outside of frustum <- \\        |
                            \\ (-w/2)|
                              o------o IMAGE PLANE
                              \\     |
                               \\    |
                                \\   |fx
                                 \\  |
                                  \\ |
                                     O PINHOLE

    Args:
        u: pixel's x-coordinate
        v: pixel's y-coordinate
        fx: focal length in x-direction, measured in pixels.
        fy: focal length in y-direction,  measured in pixels.
        img_w: image width (in pixels)
        img_h: image height (in pixels)

    Returns:
        ray_dir: direction of 3d ray, in the camera frame.
    """
    if not np.isclose(fx, fy, atol=1e-3):
        raise ValueError(f"Focal lengths in the x and y directions must match: {fx} != {fy}")

    # approximation for principal point
    px = img_w / 2
    py = img_h / 2

    # the camera coordinate frame (where Z is out, x is right, y is down).

    # compute offset from the center
    x_center_offs = u - px
    y_center_offs = v - py

    ray_dir = np.array([x_center_offs, y_center_offs, fx])
    ray_dir /= np.linalg.norm(ray_dir)
    return ray_dir


def compute_pixel_ray_directions_vectorized(uv: np.ndarray, fx: float, fy: float, img_w: int, img_h: int) -> np.ndarray:
    """Given (u,v) coordinates and intrinsics, generate pixel rays in cam. coord frame.

    Assume +z points out of the camera, +y is downwards, and +x is across the imager.

    Args:
        uv: Numpy array of shape (N,2) with (u,v) coordinates
        fx: focal length in x-direction, measured in pixels.
        fy: focal length in y-direction,  measured in pixels.
        img_w: image width (in pixels)
        img_h: image height (in pixels)

    Returns:
        ray_dirs: Array of shape (N,3) with ray directions in camera frame.
    """
    if not np.isclose(fx, fy, atol=1e-3):
        raise ValueError(f"Focal lengths in the x and y directions must match: {fx} != {fy}")

    assert uv.shape[1] == 2

    # Approximation for principal point
    px = img_w / 2
    py = img_h / 2

    u = uv[:, 0]
    v = uv[:, 1]
    num_rays = uv.shape[0]
    ray_dirs = np.zeros((num_rays, 3))
    # x center offset from center
    ray_dirs[:, 0] = u - px
    # y center offset from center
    ray_dirs[:, 1] = v - py
    ray_dirs[:, 2] = fx

    # elementwise multiplication of scalars requires last dim to match
    ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    assert ray_dirs.shape[1] == 3
    return ray_dirs
