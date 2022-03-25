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
Tesselation/triangulation utilities.

We use `mapbox earcut` for ...
We use `trimesh` for ...
"""

import mapbox_earcut as earcut
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from argoverse.utils.manhattan_search import compute_point_cloud_bbox

from shapely.geometry import Polygon

MAX_EDGE_LENGTH_TRIANGLES_M = 0.25  # meters


def form_polygon_triangulation_2d_brute_force(
    polygon_pts: np.ndarray, downsample_factor=1.0, allow_interior_only: bool = True
) -> np.ndarray:
    """Given a polygon, triangulate it at a specific resolution specified by the downsample factor.

    Polygon points must be ordered! Tesselate regular grid of triangles
    that lie within some polygon

    Smaller downsample_factor increases the resolution

    Args:
        polygon_pts: array of shape (N,2)
        downsample_factor:
        allow_interior_only:

    Returns:
        array of shape (N,6), with (v0x, v0y), (v1x, v1y), (v2x, v2y)
        vertices are arranged counterclockwise

        v2 - v3
        |  \\ |
        v0 - v1
    """
    min_x, min_y, max_x, max_y = compute_point_cloud_bbox(polygon_pts)

    nx = max_x - min_x
    ny = max_y - min_y
    x = np.linspace(min_x, max_x, int((nx + 1) / downsample_factor))
    y = np.linspace(min_y, max_y, int((ny + 1) / downsample_factor))
    x_grid, y_grid = np.meshgrid(x, y)

    num_rows = x_grid.shape[0]
    num_cols = x_grid.shape[1]

    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    x_grid = x_grid[:, np.newaxis]
    y_grid = y_grid[:, np.newaxis]

    grid_pts = np.hstack([x_grid, y_grid])

    triangles_lower = subdivide_grid_cells_into_triangles(grid_pts, num_rows, num_cols, section="lower_triangular")
    triangles_upper = subdivide_grid_cells_into_triangles(grid_pts, num_rows, num_cols, section="upper_triangular")

    triangles = np.vstack([triangles_upper, triangles_lower])

    if allow_interior_only:
        # Note using shapely for point-in-polygon tests on every triangle vertex is EXTREMELY slow.
        polygon_shapely = Polygon(polygon_pts)
        inside_poly = [polygon_shapely.contains(Polygon(tri.reshape(3, 2))) for tri in triangles]
        interior_triangles = triangles[inside_poly]
        return interior_triangles

    else:
        # return all, even if fall outside polygon
        return triangles


def subdivide_grid_cells_into_triangles(grid_pts, num_rows: int, num_cols: int, section: str):
    """Split each square grid cell into two triangles, along the diagonal.

    For lower triangular portion of grid cell:
        check validity of index pairs, x coord cannot be on far right edge
    For upper triangular portion of grid cell:
        cannot be on bottom, and cannot be on far right

    Args:
        grid_pts:
        num_rows: number of rows...
        num_cols: number of columns...
        section: portion of grid cell, either "upper_triangular", or "lower_triangular"

    Returns:
        triangles:
    """
    is_valid_x = np.ones((num_rows, num_cols), dtype=bool)
    is_valid_x[:, -1] = False
    is_valid_y = np.ones((num_rows, num_cols), dtype=bool)

    if section == "lower_triangular":
        is_valid_y[-1, :] = False
    elif section == "upper_triangular":
        is_valid_y[0, :] = False

    is_valid_x = is_valid_x.flatten()
    is_valid_y = is_valid_y.flatten()
    valid_idxs = np.logical_and(is_valid_x, is_valid_y)

    num_grid_pts = grid_pts.shape[0]
    v0_idxs = np.arange(num_grid_pts)

    if section == "lower_triangular":
        v1_idxs = v0_idxs + 1
        v2_idxs = v0_idxs + num_cols
    elif section == "upper_triangular":
        v1_idxs = v0_idxs - num_cols + 1
        v2_idxs = v0_idxs + 1

    v0_idxs = v0_idxs[valid_idxs]
    v1_idxs = v1_idxs[valid_idxs]
    v2_idxs = v2_idxs[valid_idxs]

    v0 = grid_pts[v0_idxs]
    v1 = grid_pts[v1_idxs]
    v2 = grid_pts[v2_idxs]
    triangles = np.hstack([v0.reshape(-1, 2), v1.reshape(-1, 2), v2.reshape(-1, 2)])
    return triangles


def form_polygon_triangulation_2d(
    polygon_pts: np.ndarray, downsample_factor=1.0, allow_interior_only: bool = True
) -> np.ndarray:
    """Create a triangulation of a possibly non-convex 2d polygon using Mapbox earcut bindings.

    Note: Shapely's polygon triangulation function (shapely.ops.triangulate) cannot handle
    non-convex polygons (i.e. creates a Delaunay triangulation). Instead, we use earcut
    to achieve the correct triangulation. Alternate libraries include https://github.com/lycantropos/sect.

    We subdivide the polygon afterwards in order to be able to render only triangles
    with all 3 vertices in the image, instead of computing frustum intersections for culling.

    Args:
        polygon_pts: array of shape (N,2)
        downsample_factor:
        allow_interior_only:

    Returns:
        array of shape (N,6), with (v0x, v0y), (v1x, v1y), (v2x, v2y)
        vertices are arranged counterclockwise
    """
    if not allow_interior_only:
        return form_polygon_triangulation_2d_brute_force(
            polygon_pts, downsample_factor=downsample_factor, allow_interior_only=False
        )

    N = polygon_pts.shape[0]

    # `rings` -- an array of end-indices for each ring.
    # The first ring is the outer contour of the polygon.
    # Subsequent ones are holes.
    # This implies that the last index must always be equal to the size of verts!
    rings = np.array([N])

    tri_idxs = earcut.triangulate_float32(polygon_pts[:, :2], rings)

    # import pdb; pdb.set_trace()
    tri_idxs = tri_idxs.reshape(-1, 3)
    n_triangulation = len(tri_idxs)
    triangles = np.zeros((n_triangulation, 6))
    for i, v_idxs in enumerate(tri_idxs):
        triangles[i] = polygon_pts[v_idxs, :2].reshape(6)

    triangles = np.array(triangles)
    # import pdb; pdb.set_trace()

    subdivided_triangles = subdivide_mesh(triangles, max_edge_length=MAX_EDGE_LENGTH_TRIANGLES_M)
    return subdivided_triangles


def subdivide_mesh(triangles: np.ndarray, max_edge_length: float) -> np.ndarray:
    """Subdivide a mesh until every edge is shorter than a specified length using the Trimesh library.

    Note: Will return a triangle soup, not a nicely structured mesh.

    Args:
        triangles: array of shape (N,6), with (v0x, v0y), (v1x, v1y), (v2x, v2y)
        max_edge_length: maximum length of any edge in the result.

    Returns:
        subdivided_triangles: array of shape (M,6), with (v0x, v0y), (v1x, v1y), (v2x, v2y)
    """
    tri_verts_2d = triangles.reshape(-1, 2)
    n_verts_2d = len(tri_verts_2d)
    vertices_3d = np.zeros((n_verts_2d, 3))
    vertices_3d[:, :2] = tri_verts_2d
    faces = np.arange(n_verts_2d).reshape(-1, 3)

    # `vertices` arg must be (N,3) and float array (vertices in space)
    # `faces` arg must be (M,3) and int array (indices of vertices which make up triangles)
    subd_vertices, subd_faces = trimesh.remesh.subdivide_to_size(
        vertices=vertices_3d, faces=faces, max_edge=max_edge_length, max_iter=20, return_index=False
    )

    n_subd_faces = len(subd_faces)
    subdivided_triangles = np.zeros((n_subd_faces, 6))
    for f, face_idxs in enumerate(subd_faces):
        subdivided_triangles[f] = subd_vertices[face_idxs, :2].flatten()

    return subdivided_triangles
