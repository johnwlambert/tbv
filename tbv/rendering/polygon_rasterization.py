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

import av2.utils.depth_map_utils as depth_map_utils
import numpy as np
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.map.map_api import RasterLayerType
from av2.rendering.map import EgoViewMapRenderer

import tbv.utils.cv2_img_utils as cv2_img_utils
import tbv.utils.triangulation_2d as triangulation_2d


def render_polygon_egoview(
    ego_metadata: EgoViewMapRenderer,
    img_bgr: np.ndarray,
    polygon_pts_cityfr: np.ndarray,
    downsample_factor: float,
    allow_interior_only: bool,
    filter_to_driveable_area: bool,
    color_bgr: Tuple[int, int, int],
) -> np.ndarray:
    """Triangulate a polygon in 2d, then potentially filter the triangles based on a 2d raster grid, then
    append 3d coordinates. Then project triangles into a perspective camera and rasterize.

    Args:
        ego_metadata:
        img_bgr:
        polygon_pts_cityfr:
        downsample_factor:
        allow_interior_only:
        filter_to_driveable_area:
        color_bgr: BGR 3-tuple in range [0,255].
    """
    # region not well expressed near image borders if downsample_factor is not very small
    triangles = triangulation_2d.form_polygon_triangulation_2d(
        polygon_pts_cityfr, downsample_factor, allow_interior_only
    )

    if filter_to_driveable_area:
        verts_cityfr = triangles.reshape(-1, 2)
        verts_cityfr = ego_metadata.avm.append_height_to_2d_city_pt_cloud(points_xy=verts_cityfr)
        vert_inside_da = ego_metadata.avm.get_raster_layer_points_boolean(
            verts_cityfr, layer_name=RasterLayerType.DRIVABLE_AREA
        )
        tri_inside_da = vert_inside_da.reshape(-1, 3).sum(axis=1) == 3
        da_triangles = triangles[tri_inside_da]
        triangles = da_triangles

    tri_verts_cityfr = triangles.reshape(-1, 2)
    tri_verts_cityfr = ego_metadata.avm.append_height_to_2d_city_pt_cloud(points_xy=tri_verts_cityfr)
    tri_verts_egofr = ego_metadata.city_SE3_ego.inverse().transform_point_cloud(tri_verts_cityfr)

    img_bgr = render_triangles_in_egoview(
        depth_map=ego_metadata.depth_map,
        img_bgr=img_bgr,
        tri_verts_egofr=tri_verts_egofr,
        pinhole_camera=ego_metadata.pinhole_cam,
        color_bgr=color_bgr,
    )
    return img_bgr


def render_triangles_in_egoview(
    depth_map: np.ndarray,
    img_bgr: np.ndarray,
    tri_verts_egofr: np.ndarray,
    pinhole_camera: PinholeCamera,
    color_bgr: Tuple[int, int, int],
) -> np.ndarray:
    """Rasterize triangles (provided in the egovehicle frame) onto a 2d canvas.

    Note: Uses occlusion reasoning to reason about the triangles visible from the perspective of a single camera.

    Args:
        depth_map: array of shape (H,W) representing a simulated depth map, using sparse LiDAR returns.
        img_bgr: array of shape (H,W,3) representing BGR image
        tri_verts_egofr: array of shape (N,3), arranged with triplets adjacent to one another,
            i.e. indices [0,1,2] represent one triangle, [3,4,5] also etc
        pinhole_camera:
        color_bgr: BGR 3-tuple in range [0,255].

    Returns:
        img_bgr: array of shape () representing a BGR image
    """
    initial_num_triangles = tri_verts_egofr.shape[0] // 3

    uv, points_cam, valid_pts_bool = pinhole_camera.project_ego_to_img(points_ego=tri_verts_egofr, remove_nan=False)
    tri_valid_cheirality = valid_pts_bool.reshape(-1, 3).sum(axis=1) == 3

    # find which vertices are valid according to the depth map
    allowed_noise = depth_map_utils.compute_allowed_noise_per_point(points_cam)

    n_verts = tri_verts_egofr.shape[0]
    # not occluded vs. semantic entity triangle z coord
    depth_map_z = np.ones((n_verts)) * np.nan
    uv_valid = np.round(uv[valid_pts_bool]).astype(np.uint32)
    depth_map_z[valid_pts_bool] = depth_map[uv_valid[:, 1], uv_valid[:, 0]]

    vert_is_visible = points_cam[:, 2] <= depth_map_z + allowed_noise

    all_tri_verts_visible = vert_is_visible.reshape(-1, 3).sum(axis=1) == 3

    tri_triplet_valid = np.logical_and(tri_valid_cheirality, all_tri_verts_visible)
    valid_tri_idxs = np.arange(initial_num_triangles)[tri_triplet_valid]

    for orig_tri_idx in valid_tri_idxs:
        vert_start_idx = orig_tri_idx * 3
        vert_end_idx = vert_start_idx + 3
        tri_uv = uv[vert_start_idx:vert_end_idx].astype(np.int32)
        img_bgr = cv2_img_utils.draw_polygon_cv2(tri_uv, img_bgr, color_bgr)

    return img_bgr
