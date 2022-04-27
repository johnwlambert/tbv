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
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.se3 import SE3

import tbv.utils.frustum_utils as frustum_utils


TRIANGLES_TYPE = List[Tuple[np.ndarray, np.ndarray, np.ndarray]]


def get_ground_surface_grid_triangles(
    avm: ArgoverseStaticMap, city_SE3_egovehicle: SE3, range_m: float = 30
) -> TRIANGLES_TYPE:
    """Subdivide a large flat, square area (with zero height) into triangles.

    Square area has exact area (2*range_m * 2*range_m). 1 meter resolution is used.

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
            v0 = np.array([x, y, 0]).astype(np.float32)
            v1 = np.array([x + 1, y, 0]).astype(np.float32)
            v2 = np.array([x, y + 1, 0]).astype(np.float32)
            v3 = np.array([x + 1, y + 1, 0]).astype(np.float32)

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

            v0 = np.array([x, y, 0]).astype(np.float32)
            v1 = np.array([x + 1, y, 0]).astype(np.float32)
            v2 = np.array([x, y + 1, 0]).astype(np.float32)
            v3 = np.array([x + 1, y + 1, 0]).astype(np.float32)

            tri_lower = [v0, v1, v2]
            tri_upper = [v2, v1, v3]

            nearby_triangles.append(tri_lower)
            nearby_triangles.append(tri_upper)

    return nearby_triangles


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
