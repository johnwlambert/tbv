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

import copy
import pdb
from typing import Any, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.calibration import (
    point_cloud_to_homogeneous,
    project_lidar_to_img_motion_compensated,
    project_lidar_to_img
)

from tbv.rendering.bev_sensor_utils import interp_dense_grid_from_sparse
from tbv.utils.proj_utils import within_img_bnds


MIN_DISTANCE_AWAY = 30 # assume max noise starting at this distance
MAX_ALLOWED_NOISE = 3 # meters


def compute_allowed_noise_per_point(uv_cam: np.ndarray) -> np.ndarray:
    """

    Args:
        uv_cam: array of shape (N,3) representing 3d points in the camera coordinate frame.

    Returns:
        allowed_noise:
    """
    dists_away = np.linalg.norm(uv_cam, axis=1)
    
    max_dist_away = dists_away.max()
    max_dist_away = max(max_dist_away, MIN_DISTANCE_AWAY)

    allowed_noise = (dists_away / max_dist_away) * MAX_ALLOWED_NOISE
    return allowed_noise


def get_depth_map_from_lidar(
    lidar_pts: np.ndarray,
    log_calib_data: Any,
    camera_config: Any,
    camera_name: str,
    data_dir: str,
    log_id: str,
    cam_timestamp: int,
    lidar_timestamp: int,
    img_h: int,
    img_w: int
) -> Optional[np.ndarray]:
    """

    Args:
        lidar_pts: array of shape ()
        log_calib_data: 
        camera_config: 
        camera_name: 
        data_dir: 
        log_id: unique identifier of log/scenario.
        cam_timestamp: timestamp when image was captured, measured in nanoseconds.
        lidar_timestamp: timestamp when LiDAR was captured, measured in nanoseconds.
        img_h: desired height of depth image, in pixels.
        img_w: desired width of depth image, in pixels.

    Returns:
        depth_map:
    """
    points_h = point_cloud_to_homogeneous(lidar_pts).T
    # motion compensate always
    uv, uv_cam, valid_pts_bool, K = project_lidar_to_img_motion_compensated(
        points_h,  # these are recorded at lidar_time
        copy.deepcopy(log_calib_data),
        camera_name,
        cam_timestamp,
        lidar_timestamp,
        data_dir,
        log_id,
        return_K=True,
    )
    if uv is None:
        # poses were missing for either the camera or lidar timestamp
        return None
    if valid_pts_bool.sum() == 0:
        return None
    uv_cam = uv_cam.T

    # rounding mean some points end up on boundary outside of image
    within_bnds = within_img_bnds( np.round(uv).astype(np.int32), camera_config)
    valid_pts_bool = np.logical_and.reduce(
        [
            valid_pts_bool,
            within_bnds
        ])

    u = np.round(uv[:,0][valid_pts_bool]).astype(np.int32)
    v = np.round(uv[:,1][valid_pts_bool]).astype(np.int32)
    z = uv_cam[:,2][valid_pts_bool]

    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    
    # # form depth map from LiDAR
    interp_depth_map = False # True
    if interp_depth_map:
        import pdb; pdb.set_trace()
        if u.max() > camera_config.img_width or v.max() > camera_config.img_height:
            raise RuntimeError("Regular grid interpolation will fail due to out-of-bound inputs.")

        depth_map = interp_dense_grid_from_sparse(
            bev_img = depth_map,
            points = np.hstack([ u.reshape(-1,1), v.reshape(-1,1) ]),
            rgb_values = z,
            grid_h = img_h,
            grid_w = img_w,
            interp_method = 'linear'
        )
    else:
        depth_map[v, u] = z # 255 #z

    #vis_depth_map(img_bgr, depth_map, interp_depth_map)
    return depth_map


def vis_depth_map(img_bgr: np.ndarray, depth_map: np.ndarray, interp_depth_map: bool) -> None:
    """Visualize a depth map using Matplotlib's `inferno` colormap.

    Args:
        img_bgr:
        depth_map
        interp_depth_map:
    """

    # if not interp_depth_map:
    #     # fix the zero values?
    #     depth_map[ depth_map == 0] = np.finfo(np.float32).max

    # NUM_DILATION_ITERS = 10 # 10
    # # purely for visualization
    # depth_map = cv2.dilate(
    #     depth_map,
    #     kernel=np.ones((2, 2), np.uint8),
    #     iterations=NUM_DILATION_ITERS
    # )

    # prevent too dark in the foreground, clip far away
    depth_map = np.clip(depth_map,0,50)

    plt.subplot(1,2,1)
    plt.imshow(img[:,:,::-1])
    plt.subplot(1,2,2)
    plt.imshow( (depth_map*3).astype(np.uint8), cmap='inferno')
    plt.show()
    plt.close('all')
