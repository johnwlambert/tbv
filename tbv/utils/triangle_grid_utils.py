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

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import argoverse.utils.calibration as calib_utils
from argoverse.map_representation.map_api_v2 import ArgoverseStaticMapV2
from argoverse.utils.se3 import SE3
from argoverse.utils.cv2_plotting_utils import draw_polygon_cv2

from tbv.utils.depth_map_utils import compute_allowed_noise_per_point
import tbv.utils.frustum_utils as frustum_utils

TRIANGLES_TYPE = List[Tuple[np.ndarray, np.ndarray, np.ndarray]]


def get_ground_surface_grid_triangles(
    avm: ArgoverseStaticMapV2, city_SE3_egovehicle: SE3, range_m: float = 30
) -> TRIANGLES_TYPE:
    """Subdivide a large flat, square area (with zero height) into triangles.

    Square area has exact area (2*range_m * 2*range_m).

    v2 - v3
    |  \\ |
    v0 - v1

    Vertices are arranged counterclockwise (CCW).

    Args:
        avm: vector map object.
        city_SE3_egovehicle: pose of the AV/egovehicle in the city coordinate frame.
        range_m: range, in meters, by l-infinity norm to cover with triangles.

    Returns:
        nearby_triangles: parameterization of the ground surface, as a triangle mesh,
           with vertices provided in the egovehicle frame (NOT in the city frame).
    """
    nearby_triangles = []
    # loop over x
    for x in range(-range_m, range_m):
        # loop over y
        for y in range(-range_m, range_m):

            # we set z=0 since on a flat plane.
            v0 = np.array([x, y, 0]).astype(np.float)
            v1 = np.array([x + 1, y, 0]).astype(np.float)
            v2 = np.array([x, y + 1, 0]).astype(np.float)
            v3 = np.array([x + 1, y + 1, 0]).astype(np.float)

            # fmt: off
            vertices_ego_fr = np.vstack(
                [
                    v0.reshape(1, 3),
                    v1.reshape(1, 3),
                    v2.reshape(1, 3),
                    v3.reshape(1, 3)
                ]
            )
            # fmt: on

            vertices_city_fr = city_SE3_egovehicle.transform_point_cloud(vertices_ego_fr)
            v_heights = avm.raster_ground_height_layer.get_ground_height_at_xy(vertices_city_fr)

            vertices_city_fr[:, 2] = v_heights
            egovehicle_SE3_city = city_SE3_egovehicle.inverse()
            vertices_ego_fr = egovehicle_SE3_city.transform_point_cloud(vertices_city_fr)

            v0 = vertices_ego_fr[0]
            v1 = vertices_ego_fr[1]
            v2 = vertices_ego_fr[2]
            v3 = vertices_ego_fr[3]

            tri_lower = [v0, v1, v2]
            tri_upper = [v2, v1, v3]

            nearby_triangles.append(tri_lower)
            nearby_triangles.append(tri_upper)

    return nearby_triangles


def get_flat_plane_grid_triangles(range_m: float = 30) -> TRIANGLES_TYPE:
    """
    v2 - v3
    |  \\ |
    v0 - v1

    Args:
        range_m:

    Returns:
        nearby_triangles:
    """
    nearby_triangles = []
    # loop over x
    for x in range(-range_m, range_m):
        # loop over y
        for y in range(-range_m, range_m):

            v0 = np.array([x, y, 0]).astype(np.float)
            v1 = np.array([x + 1, y, 0]).astype(np.float)
            v2 = np.array([x, y + 1, 0]).astype(np.float)
            v3 = np.array([x + 1, y + 1, 0]).astype(np.float)

            tri_lower = [v0, v1, v2]
            tri_upper = [v2, v1, v3]

            nearby_triangles.append(tri_lower)
            nearby_triangles.append(tri_upper)

    return nearby_triangles


def test_get_flat_plane_grid_triangles() -> None:
    """ """
    nearby_triangles = get_flat_plane_grid_triangles(range_m=1)
    assert len(nearby_triangles) == 8

    for range_m in range(30):
        tris = get_flat_plane_grid_triangles(range_m)
        print(f"{len(tris)} at range={range_m}")


def plane_point_side_v3(p, v):
    """Get sign of point to plane distance.
    This function does not compute the actual distance.
    Positive denotes that point v is on the same side of the plane as the plane's normal vector.
    Negative if it is on the opposite side.
    
    Args:
        p: Array of shape (3,) representing a plane in Hessian Normal Form, ax + by + c = 0
        v: A vector/2D point
    
    Returns:
        sign: A float-like value representing sign of signed distance
    """
    return p[:2].dot(v) + p[2]


def get_2d_rotmat_from_theta(theta: float) -> np.ndarray:
    """
    Args:
        theta: rotation angle in radians.

    Returns:
        array of shape (2,2) representing 2d rotation matrix, corresponding to rotation theta.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def prune_triangles_to_2d_frustum(
    triangles: TRIANGLES_TYPE, yaw: float, fov_theta: float, margin: float = 1
) -> Tuple[TRIANGLES_TYPE, np.ndarray]:
    """View frustum culling of a 3d mesh.

    TODO: maybe easier to specify with explicit fov argument?

    Args:
        allowed margin, in meters since centroid is slightly inaccurate,
            but inaccuracy is bounded by resolution of regular grid
        yaw: yaw angle, must be specified in radians
        fov_theta: angle, must be specified in radians

    Returns:
        triangles that roughly lie within the viewing frustum
        boolean whe
    """
    num_triangles = len(triangles)
    range_m = 2

    tri_centroids = [np.mean(np.array(tri), axis=0) for tri in triangles]
    tri_centroids = np.array(tri_centroids)
    # only care about (x,y) plane currently
    tri_centroids = tri_centroids[:, :2]

    # Compute normals
    l_normal, r_normal = frustum_utils.get_frustum_side_normals(fov_theta)

    # get 2d rotation matrix
    R = get_2d_rotmat_from_theta(yaw)

    # rotate rays according to rotation degrees (yaw) in x-y plane
    l_normal = R.dot(l_normal)
    r_normal = R.dot(r_normal)

    # signed distances
    sdist_to_left = tri_centroids.dot(l_normal)
    sdist_to_right = tri_centroids.dot(r_normal)

    visualize = False
    if visualize:
        inside_centroids = tri_centroids[sdist_to_left >= 0]
        outside_centroids = tri_centroids[sdist_to_left < 0]

        inside_centroids = tri_centroids[sdist_to_right >= 0]
        outside_centroids = tri_centroids[sdist_to_right < 0]

        plt.scatter(inside_centroids[:, 0], inside_centroids[:, 1], 100, color="g", marker=".")
        plt.scatter(outside_centroids[:, 0], outside_centroids[:, 1], 100, color="r", marker=".")
        plt.axis("equal")
        plt.show()

    inside_frustum = np.logical_and.reduce([sdist_to_left >= -margin, sdist_to_right >= -margin])

    frustum_triangles = [triangles[i] for i in range(num_triangles) if inside_frustum[i]]
    return frustum_triangles, inside_frustum


def render_triangles_in_egoview(
    depth_map: np.ndarray,
    img_bgr: np.ndarray,
    tri_verts_egofr: np.ndarray,
    log_calib_data,
    camera_name: str,
    color: Tuple[int, int, int],
) -> np.ndarray:
    """Uses occlusion reasoning to reason about the triangles visible.

    Args:
        depth_map
        img_bgr: BGR image
        tri_verts_egofr: array of shape (N,3), arranged with triplets adjacent to one another,
            i.e. indices [0,1,2] represent one triangle, [3,4,5] also etc
        log_calib_data:
        camera_name: 
        color: 

    Returns:
        img_bgr: array of shape () representing a BGR image
    """
    # triangles = tri_verts_egofr.copy().reshape(-1,9)

    initial_num_triangles = tri_verts_egofr.shape[0] // 3
    points_h = calib_utils.point_cloud_to_homogeneous(tri_verts_egofr).T
    uv, uv_cam, valid_pts_bool = calib_utils.project_lidar_to_img(
        points_h, log_calib_data, camera_name, return_camera_config=False, remove_nan=False
    )
    uv_cam = uv_cam.T
    tri_triplet_valid = valid_pts_bool.reshape(-1, 3).sum(axis=1) == 3
    valid_tri_idxs = np.arange(initial_num_triangles)[tri_triplet_valid]
    # valid_triangles = triangles[tri_triplet_valid]
    # print('Pruned from ', initial_num_triangles, ' to ', valid_triangles.shape[0])

    for orig_tri_idx in valid_tri_idxs:

        vert_start_idx = orig_tri_idx * 3
        vert_end_idx = vert_start_idx + 3
        tri_uv = uv[vert_start_idx:vert_end_idx].astype(np.int32)
        # semantic entity triangle z coord

        tri_uv_cam = uv_cam[vert_start_idx:vert_end_idx]
        sem_entity_cam_z = tri_uv_cam[:, 2]
        depth_map_z = depth_map[tri_uv[:, 1], tri_uv[:, 0]]

        allowed_noise = compute_allowed_noise_per_point(tri_uv_cam)
        not_occluded = sem_entity_cam_z <= depth_map_z + allowed_noise
        if not all(not_occluded):
            continue

        img_bgr = draw_polygon_cv2(tri_uv, img_bgr, color)

    return img_bgr


if __name__ == "__main__":
    """ """
    # test_prune_triangles_zero_yaw()
    # test_prune_triangles_back_frustum()
    # test_get_frustum_side_normals()

    test_form_polygon_triangulation_2d()
