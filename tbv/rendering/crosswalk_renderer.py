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
Utilities for rendering crosswalks in a bird's eye view or ego-view.

As parameterized on disk in the map, a crosswalk consists of two edges.
These will be provided to this file. Both edges should be pointing in nominally the same
direction and a pedestrian is expected to move either roughly parallel to both lines or
anti-parallel to both lines.

A real or synthetic crosswalk will be rendered as a rectangle, bounded between two long edges both
extending along the principal axis of the crosswalk. We use alternating parallel strips
of white and gray (perpendicular to the principal axis) to color the object.

Note that the two edges of the crosswalk may not have the same length, so we fit a rectangle
to them by making the length of the long sides of the rectangle no longer than the shorter
of the two input edges. In other words, fit a rectangle to a quadrilateral.

Note: `concatenate(edge1, reversed(edge2))` will always represent ordered vertices of a polygon, but
no guarantee the vertices are ordered CW or CCW (no consistent winding order).
"""

import math
from typing import Tuple

import av2.geometry.interpolate as interp_utils
import av2.geometry.polyline_utils as polyline_utils
import av2.rendering.vector as vector_rendering_utils
import matplotlib.pyplot as plt
import numpy as np
from av2.rendering.map import EgoViewMapRenderer
from shapely.geometry import LineString

import tbv.rendering.polygon_rasterization as polygon_rasterization
import tbv.utils.cv2_img_utils as cv2_img_utils


GRAY_BGR = [168, 168, 168]
WHITE_BGR = [255, 255, 255]

GRAY_RGB = [169, 169, 169]
GRAY_BGR = GRAY_RGB[::-1]

N_METERS_PER_CROSSWALK_STRIPE: float = 0.5


def render_crosshatching_bev(
    img: np.ndarray, edge1: np.ndarray, edge2: np.ndarray, resolution: float, mpl_vis: bool = False
) -> np.ndarray:
    """Given a rectangular object, with a specified principal axis, use alternating parallel strips
    of white and gray (perpendicular to the principal axis) to color the object.

    Since we use OpenCV to do the rendering, we specify colors in BGR, not RGB, order.

    Args:
        img: array of shape (H,W,3) representing RGB image canvas (map), onto which we will render a crosswalk object.
        edge1: array of shape (2,2) representing waypoint coordinates of edge polylines for crosswalk 1.
            Note: units are in pixels, not in meters.
        edge2: array of shape (2,2) representing ... as above ... but for crosswalk 2.
        resolution: float representing map image resolution, i.e. number of square meters per pixel
        mpl_vis: whether to visualize result in Matplotlib.

    Returns:
        img: array of shape (H,W,3) representing RGB or BGR image (BEV map rendering) with crosswalk rendered
           onto it. (BGR or RGB because R,G,B intensities are identical for each rendered color -- grayscale.)
    """
    assert edge1.shape == (2, 2)
    assert edge2.shape == (2, 2)
    len1 = polyline_utils.get_polyline_length(edge1)
    len2 = polyline_utils.get_polyline_length(edge2)

    xwalk_len_px = min(len1, len2)
    if len1 > xwalk_len_px:
        edge1 = clip_line_segment_to_center(edge1, xwalk_len_px)
    elif len2 > xwalk_len_px:
        edge2 = clip_line_segment_to_center(edge2, xwalk_len_px)

    n_stripes_per_meter = 1 / N_METERS_PER_CROSSWALK_STRIPE

    meter_per_px = resolution
    stripe_per_px = n_stripes_per_meter * meter_per_px

    num_stripes = int(math.ceil(stripe_per_px * xwalk_len_px))

    # split polyline into dashes (stripes)
    edge1_pts = interp_utils.interp_arc(num_stripes, points=edge1[:, :2])
    edge2_pts = interp_utils.interp_arc(num_stripes, points=edge2[:, :2])

    if mpl_vis:
        fig = plt.figure(1, figsize=(10, 10), dpi=90)
        ax = fig.add_subplot(111)

    for i in range(num_stripes - 1):

        if i % 2 == 0:
            # GAP
            color = "r" if mpl_vis else GRAY_BGR
        else:
            # CROSSWALK
            color = "y" if mpl_vis else WHITE_BGR

        # find vertices for single stripe's polygon
        a1 = edge1_pts[i]
        a2 = edge1_pts[i + 1]
        b1 = edge2_pts[i]
        b2 = edge2_pts[i + 1]

        quad = np.vstack([a1, a2, b2, b1, a1])
        cv2_img_utils.draw_polygon_cv2(quad, img, color)

        if mpl_vis:
            vector_rendering_utils.plot_polygon_patch_mpl(
                polygon_pts=quad,
                ax=ax,
                color=color,
            )

    if mpl_vis:
        plt.axis("equal")
        plt.show()

    return img

    # alternatively can get cross product w/ up gravity vector
    # then get line intersection with cross product
    # then expand until cover all points with point in polygon test


def render_crosshatching_egoview(
    ego_metadata: EgoViewMapRenderer, img_bgr: np.ndarray, edge1: np.ndarray, edge2: np.ndarray
) -> np.ndarray:
    """

    Args:
        ego_metadata
        img_bgr
        edge1: units in meters
        edge2:

    Returns:
        img_bgr:
    """
    assert edge1.shape == (2, 2)
    assert edge2.shape == (2, 2)
    len1 = polyline_utils.get_polyline_length(edge1)
    len2 = polyline_utils.get_polyline_length(edge2)

    xwalk_len_m = min(len1, len2)
    if len1 > xwalk_len_m:
        edge1 = clip_line_segment_to_center(edge1, xwalk_len_m)
    elif len2 > xwalk_len_m:
        edge2 = clip_line_segment_to_center(edge2, xwalk_len_m)

    n_stripes_per_meter = 1 / N_METERS_PER_CROSSWALK_STRIPE

    num_stripes = int(math.ceil(n_stripes_per_meter * xwalk_len_m))

    edge1_pts = interp_utils.interp_arc(t=num_stripes, points=edge1[:, :2])
    edge2_pts = interp_utils.interp_arc(t=num_stripes, points=edge2[:, :2])

    for i in range(num_stripes - 1):

        if i % 2 == 0:
            # GAP
            color_bgr = GRAY_BGR
        else:
            # CROSSWALK
            color_bgr = WHITE_BGR

        a1 = edge1_pts[i]
        a2 = edge1_pts[i + 1]
        b1 = edge2_pts[i]
        b2 = edge2_pts[i + 1]

        quad = np.vstack([a1, a2, b2, b1, a1])
        # 10 cm resolution for triangle grid
        img_bgr = polygon_rasterization.render_polygon_egoview(
            ego_metadata,
            img_bgr,
            polygon_pts_cityfr=quad,
            downsample_factor=0.1,
            allow_interior_only=True,
            filter_to_driveable_area=False,
            color_bgr=color_bgr,
        )

    return img_bgr


def render_rectangular_crosswalk_bev(img: np.ndarray, edge1: np.ndarray, edge2: np.ndarray, resolution) -> np.ndarray:
    """Given two edges of a crosswalk along its principal axis, render a crosswalk as alternating white/gray strips.

    Args:
        img: array of shape (H,W,3) representing RGB image canvas (map), onto which we will render a crosswalk object.
        edge1: array of shape (2,2) representing waypoint coordinates of edge polylines for crosswalk 1.
            Note: units are in pixels, not in meters.
        edge2: array of shape (2,2) representing ... as above ... but for crosswalk 2.
        resolution: float representing map image resolution, i.e. number of square meters per pixel

    Returns:
        img: array of shape (H,W,3) representing RGB image (BEV map rendering) with crosswalk rendered
           onto it.
    """
    allowed_dtypes = [np.float32, np.float64]
    if edge1.dtype not in allowed_dtypes or edge2.dtype not in allowed_dtypes:
        raise ValueError("Waypoint coordinates of crosswalk edge polylines should be floating point numbers.")

    # fit a rectangle
    v0, v1, v2, v3 = get_rectangular_region(edge1, edge2)

    mpl_vis = False
    if mpl_vis:
        fig = plt.figure(1, figsize=(10, 10), dpi=90)
        ax = fig.add_subplot(111)

        color = "y"
        quad = np.vstack([v0, v1, v3, v2])
        vector_rendering_utils.plot_polygon_patch_mpl(
            polygon_pts=quad,
            ax=ax,
            color=color,
        )
        plt.axis("equal")
        plt.show()

    v0 = v0.reshape(1, 2)
    v1 = v1.reshape(1, 2)
    v2 = v2.reshape(1, 2)
    v3 = v3.reshape(1, 2)

    edge1_rect = np.vstack([v0, v1])
    edge2_rect = np.vstack([v2, v3])

    img = render_crosshatching_bev(img, edge1=edge1_rect, edge2=edge2_rect, resolution=resolution)
    return img


def get_rectangular_region(
    edge1: np.ndarray, edge2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a rectangular region given two roughly parallel edges, potentially of different lengths.

    Try first:
        v2 ----> v3 (line b)
        //\\
         || Normal
         ||
        v0 ----> v1 (line a)

    Then try:
        v0 ----> v1 (line a)
         || Normal
         ||
        \\//
        v2 ----> v3 (line b)

    Both lines should be pointing in nominally the same direction and a pedestrian is expected to
    move either roughly parallel to both lines or anti-parallel to both lines.

    Args:
        edge1: array of shape (2,2) representing waypoint coordinates of edge polylines for crosswalk 1.
        edge2: array of shape (2,2) representing ... for crosswalk 2.

    Returns:
        v0: vertex 0, array of shape (2,) startpoint of new edge1 (long side of rectangle).
           Corresponds to input edge that was shorter side.
        v1: vertex 1, array of shape (2,) endpoint of new edge1
        v2: vertex 2, array of shape (2,) startpoint of new edge2 (other long side of rectangle)
        v3: vertex 3, array of shape (2,) endpoint of new edge2
    """
    assert edge1.shape == (2, 2)
    assert edge2.shape == (2, 2)
    len1 = polyline_utils.get_polyline_length(edge1)
    len2 = polyline_utils.get_polyline_length(edge2)

    # choose the shortest length as our width
    if len1 > len2:
        line_a = edge2
        line_b = edge1
    else:
        line_a = edge1
        line_b = edge2

    # line_a will be the short edge, and line_b will be the long edge
    v0, v1 = line_a

    # get vector going from v0 -> v1 on line_a
    a_normal = compute_line_segment_normal(line_a)
    a_normal *= 10000  # MAX_LEN --> guarantee some intersection, if exists

    # get unit direction of line b
    b_vec = np.diff(line_b, axis=0).squeeze()
    b_vec /= np.linalg.norm(b_vec)
    b_vec *= 10000  # MAX_LEN --> guarantee some intersection, if exists

    # try going from start of line_a in direction of its normal
    ls1 = LineString([list(line_a[0]), list(line_a[0] + a_normal)])
    ls2 = LineString([line_b[0] - b_vec, line_b[0] + b_vec])
    v2 = list(ls1.intersection(ls2).coords)

    if len(v2) == 0:
        # if unsuccessful, try going from start of line_a in opposite direction of its normal
        # (in the other orthogonal direction instead)
        ls1 = LineString([list(line_a[0]), list(line_a[0] - a_normal)])
        ls2 = LineString([line_b[0] - b_vec, line_b[0] + b_vec])
        v2 = list(ls1.intersection(ls2).coords)

    if len(v2) == 0:
        raise RuntimeError("Invalid crosswalk parameterization, no rectangular region could be fit.")

    v2 = np.array(v2).squeeze()

    # make new line_b have the exact same direction and length as line_a
    line_a_dir = np.diff(line_a, axis=0).squeeze()
    v3 = v2 + line_a_dir
    return v0.squeeze(), v1.squeeze(), v2, v3.squeeze()


def compute_line_segment_normal(ls: np.ndarray) -> np.ndarray:
    """Returns normal by lifting 2d line segment to 3d, taking crossproduct with upright unit vector.

    Args:
       ls: array of shape (2,2) representing two endpoints of a line segment in xy plane.

    Returns:
       ortho_vec: array of shape (2,) representing unit-length line normal vector.
    """
    # compute direction vector going from start to end of line segment
    line_dir = np.diff(ls, axis=0).squeeze()
    # In 3d space, take cross-product with standard basis vector e3
    normal = np.cross(np.array([0, 0, 1]), np.array([line_dir[0], line_dir[1], 0]))
    normal = normal.squeeze()
    # normalize to unit length
    normal /= np.linalg.norm(normal)
    return normal[:2]


def clip_line_segment_to_center(line_segment: np.ndarray, clip_len: float):
    """Clip a line segment to a specified length, by trimming an equal amount from each end.

    Effectively shrinks the polyline away from endpoints towards center.
    We try keeping all of the points, and if the requested length parameter is not
    satisfied, we remove points one by one from each end

    Args:
        line_segment: array of shape (N,2), which may need to be clipped.
        clip_len: desired length of line segment, after clipping.
    """
    original_len = polyline_utils.get_polyline_length(line_segment)
    N_PTS = 100
    pts = interp_utils.interp_arc(t=N_PTS, points=line_segment[:, :2])

    for i in range(1, N_PTS // 2):
        # remove one more point at a time, from both sides
        new_len = polyline_utils.get_polyline_length(pts[i:-i])
        if new_len <= clip_len:
            # print(f"Clipped from {original_len:.1f} m. to {new_len:.1f} m., to satisfy {clip_len:.1f} m.") # noqa
            return pts[i:-i]

    raise RuntimeError("Error in `clip_line_segment_to_center().`")


def render_rectangular_crosswalk_egoview(
    ego_metadata: EgoViewMapRenderer, img_bgr: np.ndarray, edge1: np.ndarray, edge2: np.ndarray
) -> np.ndarray:
    """

    Args:
        ego_metadata: metadata required for rendering 3d data into a single view frustum (the "ego-view").
        img_bgr: array of shape ()
        edge1:
        edge2:

    Returns:
        img_bgr:
    """
    assert edge1.dtype in [np.float32, np.float64]
    assert edge2.dtype in [np.float32, np.float64]
    v0, v1, v2, v3 = get_rectangular_region(edge1, edge2)

    v0 = v0.reshape(1, 2)
    v1 = v1.reshape(1, 2)
    v2 = v2.reshape(1, 2)
    v3 = v3.reshape(1, 2)

    img_bgr = render_crosshatching_egoview(ego_metadata, img_bgr, edge1=np.vstack([v0, v1]), edge2=np.vstack([v2, v3]))
    return img_bgr
