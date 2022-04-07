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
Utilities for finding correspondences between the image plane and the ground plane.
This can be done by using known 3d point locations, and projecting them into the image (e.g. LiDAR)
or by sending out rays from the image aginst a ground surface mesh (e.g. from the raster ground height map).
"""

import copy
import logging
import time
from pathlib import Path
from typing import Any, Tuple

import av2.geometry.mesh_grid as mesh_grid_utils
import imageio
import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.camera.pinhole_camera import PinholeCamera

import tbv.utils.logger_utils as logger_utils
import tbv.utils.mseg_interface as mseg_interface
import tbv.utils.z1_egovehicle_mask_utils as z1_mask_utils
from tbv.rendering_config import BevRenderingConfig

try:
    import tbv_raytracing
except:
    print("GPU raycasting library was not compiled, skipping import.")


logger = logging.getLogger(__name__)

NO_HIT_FLAG = -99999


def get_point_rgb_correspondences_raytracing(
    nearby_triangles: np.ndarray,
    label_map: np.ndarray,
    cam_timestamp: int,
    rgb_img: np.ndarray,
    city_SE3_egovehicle: SE3,
    pinhole_camera: PinholeCamera,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cast a ray from the camera optical center through every image pixel, until it hits the ground surface mesh.
    Record the coordinates for each intersection point, and associate an RGB value with it.
    These will later be used for rendering ground surface imagery.

    Rays in the camera frame are converted into rays in the egovehicle coordinate frame via SE(3).

    Args:
        nearby_triangles: array of shape () representing the parameterization of the ground surface,
            as a triangle mesh, with vertices provided in the egovehicle frame (NOT in the city frame).
        label_map: array of shape (H,W) representing a semantic segmentation label map.
        cam_timestamp: integer timestamp in nanoseconds of ...
        rgb_img: array of shape (H,W,3) representing an RGB image.
        city_SE3_egovehicle: pose of the egovehicle within the city coordinate frame.
        pinhole_camera: Pinhole camera object, w/ camera intrinsics + extrinsics.

    Returns:
        city_pts: array of shape (N,3) representined 3d coordinates.
        rgb_vals: array of shape (N,3) representing uint values in the range [0,255]
    """
    egovehicle_SE3_camera = pinhole_camera.ego_SE3_cam
    camera_name = pinhole_camera.cam_name

    origin = egovehicle_SE3_camera.translation.squeeze()
    img_h, img_w = rgb_img.shape[:2]
    fx = pinhole_camera.intrinsics.fx_px
    fy = pinhole_camera.intrinsics.fy_px

    rgb_vals = np.zeros((0, 3), dtype=np.uint8)
    ego_pts = np.zeros((0, 3), dtype=np.float32)

    # pass in meshgrid coords, only loop over (u,v) that are returned
    uv = mesh_grid_utils.get_mesh_grid_as_point_cloud(min_x=0, max_x=img_w - 1, min_y=0, max_y=img_h - 1)

    if label_map is not None:
        valid_semantics = mseg_interface.filter_by_semantic_classes(label_map, copy.deepcopy(uv).astype(np.int32))
        uv_valid = uv[valid_semantics].astype(np.int32)
    else:
        uv_valid = uv.astype(np.int32)

    # now filter using the egovehicle foreground masks
    if camera_name in ["ring_front_center", "ring_rear_right", "ring_rear_left"]:
        nonforegound = z1_mask_utils.filter_out_egovehicle(uv_valid, camera_name)
        uv_valid = uv_valid[nonforegound]

    # TODO: pass in all pre-computed ray directions, and then prune based on uv index
    # ray directions in the camera coordinate frame
    rays_camera = pinhole_camera.compute_pixel_ray_directions(uv_valid)
    # these rays are all facing down the camera barrel
    # Need to rotate them back into the egovehicle's coordinate system
    egovehicle_SO3_camera = SE3(rotation=egovehicle_SE3_camera.rotation, translation=np.zeros(3))
    # triangles are in the egovehicle frame, so rays must also be transformed into the egovehicle frame.
    rays_egovehicle = egovehicle_SO3_camera.transform_point_cloud(rays_camera)

    if rays_egovehicle.shape == (0, 3):
        # return empty city_pts, rgb_vals
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    start = time.time()

    num_nearby_triangles = len(nearby_triangles)
    triangle_matrix = np.zeros((num_nearby_triangles, 9))
    for i, tri in enumerate(nearby_triangles):
        v0, v1, v2 = tri
        triangle_matrix[i, :3] = v0
        triangle_matrix[i, 3:6] = v1
        triangle_matrix[i, 6:] = v2

    # print('Num triangles:', num_nearby_triangles)
    # print('Num rays:', rays_egovehicle.shape)
    num_rays = rays_egovehicle.shape[0]
    # put triangles into egovehicle frame, get ones within 30 meters away

    # initialize an empty buffer to be populated
    hits = np.zeros((num_rays, 3))
    tbv_raytracing.intersect_rays_with_tri_mesh(
        copy.deepcopy(triangle_matrix), copy.deepcopy(origin), copy.deepcopy(rays_egovehicle), hits
    )

    inter_exists_arr = get_intersection_found_logicals(hits[:, 0])
    inter_exists_arr1 = get_intersection_found_logicals(hits[:, 1])
    inter_exists_arr2 = get_intersection_found_logicals(hits[:, 2])

    inter_points_arr = copy.deepcopy(hits)

    # inter_exists_arr_eigen, inter_points_arr_eigen = raytracing.intersect_rays_with_tri_mesh(triangle_matrix, origin, rays_egovehicle)

    uv_w_intersect = uv_valid[inter_exists_arr]
    intersect_RGB = rgb_img[uv_w_intersect[:, 1], uv_w_intersect[:, 0]]
    Ps = inter_points_arr[inter_exists_arr]
    ego_pts = np.vstack([ego_pts, Ps])
    rgb_vals = np.vstack([rgb_vals, intersect_RGB])

    end = time.time()
    duration = end - start
    logger.info(f"Camera rays traced in {duration:.2f} sec for mesh.")

    rgb_vals = np.array(rgb_vals)
    ego_pts = np.array(ego_pts)
    city_pts = city_SE3_egovehicle.transform_point_cloud(ego_pts)
    return city_pts, rgb_vals


def get_intersection_found_logicals(hits_column: np.ndarray) -> np.ndarray:
    """ """
    EPS = 1e-3
    return np.abs(hits_column - NO_HIT_FLAG) > EPS


def get_point_rgb_correspondences_lidar(
    avm: ArgoverseStaticMap,
    config: BevRenderingConfig,
    label_map: np.ndarray,
    log_id: str,
    loader: AV2SensorDataLoader,
    cam_timestamp: int,
    lidar_timestamp: int,
    lidar_pts: np.ndarray,
    camera_name: str,
    rgb_img: np.ndarray,
    city_SE3_egovehicle: SE3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project LiDAR into the image. Get 2d->3d correspondences.

    Args:
        avm:
        config:
        label_map: array of shape (H,W) representing a semantic segmentation label map.
        log_id: string representing unique identifier for TbV log/scenario.
        loader: data loader.
        cam_timestamp: integer timestamp in nanoseconds of ...
        lidar_timestamp: integer timestamp in nanoseconds of ...
        lidar_pts: array of shape ()
        rgb_img: array of shape (H,W,3) representing an RGB image.
        city_SE3_egovehicle: pose of the egovehicle within the city coordinate frame.

    Returns:
        city_pts_valid: array of shape (N,3) representined 3d coordinates.
        rgb_vals: array of shape (N,3) representing uint values in the range [0,255]
    """
    uv, points_cam, is_valid = loader.project_ego_to_img_motion_compensated(
        points_lidar_time=lidar_pts,
        cam_name=camera_name,
        cam_timestamp_ns=cam_timestamp,
        lidar_timestamp_ns=lidar_timestamp,
        log_id=log_id,
    )

    # round first so that we dont get out of bounds by one later
    # convert to int before indexing into label map
    uv = np.round(uv).astype(np.int32)

    # TODO: use logical_and.reduce
    uv_valid = uv[is_valid]
    lidar_pts_valid = lidar_pts[is_valid]

    if config.filter_ground_with_semantics and label_map is not None:
        valid_semantics = mseg_interface.filter_by_semantic_classes(label_map, uv_valid)
        uv_valid = uv_valid[valid_semantics]
        lidar_pts_valid = lidar_pts_valid[valid_semantics]

    elif config.filter_ground_with_map:
        # use the map for the classification
        lidar_pts_valid_city = city_SE3_egovehicle.transform_point_cloud(lidar_pts_valid)
        is_ground = avm.raster_ground_height_layer.get_ground_points_boolean(points_xyz=lidar_pts_valid_city)

        uv_valid = uv_valid[is_ground]
        lidar_pts_valid = lidar_pts_valid[is_ground]

    rgb_vals = rgb_img[uv_valid[:, 1], uv_valid[:, 0]]

    city_pts_valid = city_SE3_egovehicle.transform_point_cloud(lidar_pts_valid)
    return city_pts_valid, rgb_vals


def filter_to_ground_projected_pixels(
    lidar_pts: np.ndarray,
    loader: AV2SensorDataLoader,
    log_id: str,
    lidar_timestamp: int,
    label_maps_dir: Path,
) -> np.ndarray:
    """

    Args:
        lidar_pts: array of shape (N,3) representing 3d coordinates of LiDAR returns.
        loader: data loader.
        log_id: string representing unique identifier for TbV log/scenario.
        lidar_timestamp: integer timestamp in nanoseconds of ...
        label_maps_dir: directory root for where semantic segmentation label maps are saved on disk.

    Returns:
        valid_ground_pts: array of shape ()
    """
    num_lidar_pts = lidar_pts.shape[0]

    # start w/ assumption that the point is invalid
    valid_ground_pts = np.zeros(num_lidar_pts, dtype=bool)

    # for each camera frustum
    for camera_enum in RingCameras:
        camera_name = camera_enum.value
        im_fpath = loader.get_closest_im_fpath(log_id, camera_name, lidar_timestamp)
        if im_fpath is None:
            continue

        img_fname_stem = Path(im_fpath).stem
        cam_timestamp = Path(im_fpath).stem.split("_")[-1]
        cam_timestamp = int(cam_timestamp)

        uv, points_cam, valid_pts_bool = loader.project_ego_to_img_motion_compensated(
            points_lidar_time=copy.deepcopy(lidar_pts),
            cam_name=camera_name,
            cam_timestamp_ns=cam_timestamp,
            lidar_timestamp_ns=lidar_timestamp,
            log_id=log_id,
        )

        # round first so that we dont get out of bounds by one later
        # convert to int before indexing into label map
        uv = np.round(uv).astype(np.int32)

        within_bnds = within_img_bnds(uv, camera_config)
        valid_projection = np.logical_and.reduce([valid_proj, within_bnds])

        # TODO: use logical_and.reduce
        uv_valid = uv[valid_projection]

        label_map_path = mseg_interface.get_mseg_label_map_fpath_from_image_info(
            label_maps_dir, log_id, camera_name, img_fname_stem
        )
        label_map = imageio.imread(label_map_path)

        valid_semantics = mseg_interface.filter_by_semantic_classes(label_map, uv_valid)

        valid_semantics_idxs = np.where(valid_semantics)[0]
        valid_camera_ground_idxs = np.where(valid_projection)[0][valid_semantics_idxs]

        camera_infers_ground = np.zeros(num_lidar_pts, dtype=bool)
        camera_infers_ground[valid_camera_ground_idxs] = 1

        valid_ground_pts = np.logical_or.reduce([valid_ground_pts, camera_infers_ground])

    return valid_ground_pts
