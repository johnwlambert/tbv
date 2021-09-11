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

import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.manhattan_search import compute_point_cloud_bbox
from argoverse.utils.mpl_plotting_utils import plot_lane_segment_patch
from shapely.geometry import Polygon


def form_polygon_triangulation_2d(
    polygon_pts: np.ndarray,
    downsample_factor = 1.0,
    allow_interior_only: bool = True
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
        polygon_shapely = Polygon(polygon_pts)
        inside_poly = [polygon_shapely.contains(Polygon(tri.reshape(3,2))) for tri in triangles]
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
    is_valid_x = np.ones((num_rows,num_cols), dtype=bool)
    is_valid_x[:,-1] = False
    is_valid_y = np.ones((num_rows,num_cols), dtype=bool)
    
    if section == "lower_triangular":
        is_valid_y[-1,:] = False
    elif section == "upper_triangular":
        is_valid_y[0,:] = False

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
    triangles = np.hstack([ v0.reshape(-1,2), v1.reshape(-1,2), v2.reshape(-1, 2) ])
    return triangles


def test_form_polygon_triangulation_2d() -> None:
    """ """
    # in-order
    polygon = np.array(
        [
            [1,1],
            [1,2],
            [2,3],
            [3,3],
            [3,2],
            [4,2],
            [3,1]
        ])
    triangles = form_polygon_triangulation_2d(polygon)
    gt_triangles = np.array(
        [
            [1., 2., 2., 1., 2., 2.],
            [2., 2., 3., 1., 3., 2.],
            [2., 3., 3., 2., 3., 3.],
            [1., 1., 2., 1., 1., 2.],
            [2., 1., 3., 1., 2., 2.],
            [2., 2., 3., 2., 2., 3.]
        ])
    assert np.allclose(gt_triangles, triangles)

    # try higher resolution
    triangles = form_polygon_triangulation_2d(polygon, 1.0)

    visualize = False
    if not visualize:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plot_lane_segment_patch(
        polygon_pts=polygon,
        ax=ax,
        color='b' #np.random.rand(3)
    )

    for tri in triangles:
        polygon_pts = tri.reshape(3,2)
        polygon_pts = np.vstack([ polygon_pts, polygon_pts[0] ])
        plot_lane_segment_patch(
            polygon_pts=polygon_pts,
            ax=ax,
            color='r' #np.random.rand(3)
        )
    plt.scatter(0,0, 10, marker='.', color='r')
    plt.show()
