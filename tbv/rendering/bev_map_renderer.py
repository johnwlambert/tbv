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
Uses OpenCV to rasterize polylines and polygons to a raster canvas, representing
the bird's eye view (BEV).
"""

import logging
import math
from pathlib import Path
from typing import Tuple

import av2.geometry.interpolate as interp_utils
import av2.geometry.polyline_utils as polyline_utils
import cv2
import numpy as np
from av2.geometry.mesh_grid import get_mesh_grid_as_point_cloud
from av2.geometry.sim2 import Sim2
from av2.map.lane_segment import LaneMarkType, LaneSegment
from av2.map.map_api import RasterLayerType

import tbv.rendering.crosswalk_renderer as crosswalk_renderer
import tbv.utils.cv2_img_utils as cv2_img_utils
from tbv.common.local_vector_map import LocalVectorMap


WPT_INFTY_NORM_INTERP_NUM = 50

MARKING_DELETION_PATH_LENGTH = 3
MARKING_COLOR_CHANGE_PATH_LENGTH = 3
BIKE_LANE_DESIRED_PATH_LENGTH = 5

# dark gray used for drivable area color
DARK_GRAY_INTENSITY = 100

WHITE_RGB = [255, 255, 255]
WHITE_BGR = WHITE_RGB[::-1]
YELLOW_RGB = [255, 255, 0]
YELLOW_BGR = YELLOW_RGB[::-1]

# red represents an implicit lane marking (with no actual paint, merely inferred by road users)
RED_RGB = [255, 0, 0]
RED_BGR = RED_RGB[::-1]

TRAFFIC_YELLOW1_RGB = [250, 210, 1]
TRAFFIC_YELLOW1_BGR = TRAFFIC_YELLOW1_RGB[::-1]
TRAFFIC_YELLOW2_RGB = [247, 181, 0]
TRAFFIC_YELLOW2_BGR = TRAFFIC_YELLOW2_RGB[::-1]

GRAY_RGB = [169, 169, 169]
GRAY_BGR = GRAY_RGB[::-1]

BLACK_INTENSITY = 0


# for a 2000x2000 pixel image, 6 pixels is a reason line width.
DEFAULT_LINE_WIDTH_PX = 6
DEFAULT_RESOLUTION_METERS_PER_PX = 0.02


def get_line_width_by_resolution(resolution: float) -> int:
    """Compute an appropriate polyline width, in pixels, for a specific rendering resolution.

    Note: this is not dependent upon image size -- solely dependent upon image resolution.
    Can have a tiny image at high resolution.

    Args:
        resolution: meters/pixel.

    Returns:
        line_width: line width (thickness) in pixels to use for rendering polylines with OpenCV. Must be an integer.
    """
    scale = resolution / DEFAULT_RESOLUTION_METERS_PER_PX

    # larger scale means lower resolution, so we make the line width more narrow.
    line_width = DEFAULT_LINE_WIDTH_PX / scale

    line_width = round(line_width)
    # must be at least 1 pixel thick.
    return max(line_width, 1)


def test_get_line_width_by_resolution() -> None:
    """Ensure polyline thickness is computed properly."""
    import pdb

    pdb.set_trace()
    line_width = get_line_width_by_resolution(resolution=0.02)
    assert line_width == 6
    assert isinstance(line_width, int)

    line_width = get_line_width_by_resolution(resolution=0.04)
    assert line_width == 3
    assert isinstance(line_width, int)

    line_width = get_line_width_by_resolution(resolution=0.12)
    assert line_width == 1
    assert isinstance(line_width, int)


def bev_render_window(
    lvm: LocalVectorMap,
    save_fpath: Path,
    log_id: str,
    window_center: np.ndarray,
    resolution_m_per_px: float,
    range_m: float,
) -> np.ndarray:
    """Rasterize a vector map within a particular square window of a city.

    First, we create a regular grid over the image, determine which grid points fall within the
    drivable area, and paint them gray.

    Next, we render lane polygons, and then lane markings, and then finally, pedestrian crossings.

    Args:
        lvm:
        save_fpath: absolute file path where BEV image / rendering / rasterization should be saved.
        log_id: string representing unique identifier for TbV log/scenario.
        window_center: Numpy array of shape 1x2
        resolution_m_per_px: in meters/px
        range_m: maximum range from egovehicle to consider for rendering (by l-infinity norm).
        line_width: width (in pixels) to use when rendering each polyline onto the BEV image.

    Returns:
        bev_img: image w/ BGR colors imprinted. Up on image will also be North,
          since we flipped vertically (bc of positive y-axis mirroring)
    """
    line_width = get_line_width_by_resolution(resolution_m_per_px)

    xcenter, ycenter = window_center.astype(np.int32).squeeze()

    resolution_px_per_m = 1 / resolution_m_per_px

    n_px = int(range_m * resolution_px_per_m)
    grid_h = n_px * 2
    grid_w = n_px * 2
    bev_img = np.ones((grid_h + 1, grid_w + 1, 3), dtype=np.uint8) * BLACK_INTENSITY
    img_h, img_w, _ = bev_img.shape

    # still operating in meters now, before shift to image space and scale
    # ymax is `ycenter + range_m` and xmax is `xcenter + range_m`
    ymin = ycenter - range_m
    xmin = xcenter - range_m

    # ----- Render the drivable area --------------------------------------------------
    image_Sim2_city = Sim2(R=np.eye(2), t=np.array([-xmin, -ymin]), s=resolution_px_per_m)
    city_Sim2_image = image_Sim2_city.inverse()
    grid_pts_img = get_mesh_grid_as_point_cloud(min_x=0, max_x=grid_w, min_y=0, max_y=grid_h)

    # must account for different resolution (scale down from high-res image to city coords.)
    grid_pts_city = city_Sim2_image.transform_from(grid_pts_img)
    inside_da = lvm.avm.get_raster_layer_points_boolean(grid_pts_city, layer_name=RasterLayerType.DRIVABLE_AREA)
    da_pts_city = grid_pts_city[inside_da]

    # account back for different resolution (scale up)
    da_pts_img = image_Sim2_city.transform_point_cloud(da_pts_city)
    da_pts_img = np.round(da_pts_img).astype(np.int32)
    bev_img[da_pts_img[:, 1], da_pts_img[:, 0]] = np.ones(3) * DARK_GRAY_INTENSITY

    # ---- Draw the lane polygons first -----------------------------------------------
    for _, vls in lvm.nearby_lane_segment_dict.items():
        img_bndry = image_Sim2_city.transform_from(vls.polygon_boundary[:, :2])
        cv2_img_utils.draw_polygon_cv2(img_bndry, bev_img, color=GRAY_BGR)

    for _, vls in lvm.nearby_lane_segment_dict.items():
        if vls.render_r_bound:
            bev_img = render_lane_boundary(
                bev_img=bev_img,
                vls=vls,
                side="right",
                resolution_m_per_px=resolution_m_per_px,
                image_Sim2_city=image_Sim2_city,
                line_width=line_width,
            )
        if vls.render_l_bound:
            bev_img = render_lane_boundary(
                bev_img=bev_img,
                vls=vls,
                side="left",
                resolution_m_per_px=resolution_m_per_px,
                image_Sim2_city=image_Sim2_city,
                line_width=line_width,
            )

    STOPLINE_WIDTH = 0.5  # meters
    stopline_width_px = int(math.ceil(STOPLINE_WIDTH / resolution_m_per_px))
    for stopline_city in lvm.stoplines:
        stopline_img = image_Sim2_city.transform_from(stopline_city)
        draw_polyline_cv2(stopline_img, bev_img, WHITE_BGR, img_h, img_w, thickness_px=stopline_width_px)

    # -- Draw the crosswalks last, otherwise looks weird with stoplines on top of crosswalks ----
    for lpc in lvm.ped_crossing_edges:
        # render local ped crossings (lpc's)
        (edge1_city, edge2_city) = lpc.get_edges()
        edge1_img = image_Sim2_city.transform_from(edge1_city)
        edge2_img = image_Sim2_city.transform_from(edge2_city)
        # draw_polyline_cv2(img_edge1, bev_img, xwalk_color, img_h, img_w, thickness=line_width)
        crosswalk_renderer.render_rectangular_crosswalk_bev(
            bev_img, edge1_img.astype(np.float32), edge2_img.astype(np.float32), resolution_m_per_px
        )

    bev_img = np.flipud(bev_img)
    # image is in BGR format. OpenCV cannot accept PosixPath as arg (only string.)
    cv2.imwrite(str(save_fpath), bev_img)
    return bev_img


def render_lane_boundary(
    bev_img: np.ndarray,
    vls: LaneSegment,
    side: str,
    resolution_m_per_px: float,
    image_Sim2_city: Sim2,
    line_width: float,
) -> np.ndarray:
    """Draw left or right lane boundary (only one of the two is rendered in this subroutine).

    Regarding lane marking patterns:
    Double lines are to be understood from the inside out, e.g. DASHED_SOLID means that the dashed line is adjacent
    to the lane carrying the property and the solid line is adjacent to the neighbor lane.

    Args:
        bev_img: array of shape (H,W,3) representing RGB raster canvas to render objects onto.
        vls: vector lane segment object
        side: string representing the side of the lane segment to paint, i.e. should be "left" or "right"
        resolution_m_per_px: resolution, as a fraction (number of meters per pixel).
        image_Sim2_city: Sim(2) object that maps city coordinates to image coordinates.
        line_width: width (in pixels) to use when rendering each polyline onto the BEV image.

    Returns:
        bev_img: array of shape (H,W,3) representing BGR raster canvas after objects have been rendered onto it.
    """
    img_h, img_w, _ = bev_img.shape

    if side == "right":
        polyline = vls.right_lane_boundary.xyz
        mark_type = vls.right_mark_type
    elif side == "left":
        polyline = vls.left_lane_boundary.xyz
        mark_type = vls.left_mark_type
    else:
        raise RuntimeError("Invalid `side` argument in `render_lane_boundary()`.")

    N_INTERP_PTS = 100
    # interpolation needs to happen before rounded to integer coordinates
    polyline_city = interp_utils.interp_arc(t=N_INTERP_PTS, points=polyline[:, :2])
    img_polyline = image_Sim2_city.transform_from(polyline_city)

    resolution_px_per_m = 1 / resolution_m_per_px

    # every 1 meter for now
    dash_interval_m = 1  # meter
    dash_interval_px = dash_interval_m * resolution_px_per_m

    if "WHITE" in mark_type:
        bound_color = WHITE_BGR
    elif "YELLOW" in mark_type:
        bound_color = TRAFFIC_YELLOW1_BGR
    else:
        bound_color = RED_BGR

    if ("DOUBLE" in mark_type) or ("SOLID_DASH" in mark_type) or ("DASH_SOLID" in mark_type):
        left, right = polyline_utils.get_double_polylines(img_polyline, width_scaling_factor=4)  # in pixels

    if mark_type in [LaneMarkType.SOLID_WHITE, LaneMarkType.SOLID_YELLOW, LaneMarkType.NONE]:
        draw_polyline_cv2(img_polyline, bev_img, bound_color, img_h, img_w, thickness_px=line_width)

    elif mark_type in [LaneMarkType.DOUBLE_DASH_YELLOW, LaneMarkType.DOUBLE_DASH_WHITE]:
        draw_dashed_polyline(
            left, bev_img, bound_color, img_h, img_w, thickness_px=line_width * 2, dash_interval_px=dash_interval_px
        )
        draw_dashed_polyline(
            right, bev_img, bound_color, img_h, img_w, thickness_px=line_width * 2, dash_interval_px=dash_interval_px
        )

    elif mark_type in [LaneMarkType.DOUBLE_SOLID_YELLOW, LaneMarkType.DOUBLE_SOLID_WHITE]:
        draw_polyline_cv2(left, bev_img, bound_color, img_h, img_w, thickness_px=line_width * 2)
        draw_polyline_cv2(right, bev_img, bound_color, img_h, img_w, thickness_px=line_width * 2)

    elif mark_type in [LaneMarkType.DASHED_WHITE, LaneMarkType.DASHED_YELLOW]:
        draw_dashed_polyline(
            img_polyline, bev_img, bound_color, img_h, img_w, thickness_px=line_width, dash_interval_px=dash_interval_px
        )

    elif (mark_type == LaneMarkType.SOLID_DASH_YELLOW and side == "right") or (
        mark_type == LaneMarkType.DASH_SOLID_YELLOW and side == "left"
    ):
        draw_polyline_cv2(left, bev_img, bound_color, img_h, img_w, thickness_px=line_width * 2)
        draw_dashed_polyline(
            right, bev_img, bound_color, img_h, img_w, thickness_px=line_width * 2, dash_interval_px=dash_interval_px
        )

    elif (mark_type == LaneMarkType.SOLID_DASH_YELLOW and side == "left") or (
        mark_type == LaneMarkType.DASH_SOLID_YELLOW and side == "right"
    ):
        draw_dashed_polyline(
            left, bev_img, bound_color, img_h, img_w, thickness_px=line_width * 2, dash_interval_px=dash_interval_px
        )
        draw_polyline_cv2(right, bev_img, bound_color, img_h, img_w, thickness_px=line_width * 2)

    else:
        raise ValueError(f"Unknown marking type {mark_type}")

    return bev_img


def draw_polyline_cv2(
    line_segments_arr: np.ndarray,
    image: np.ndarray,
    color: Tuple[int, int, int],
    im_h: int,
    im_w: int,
    thickness_px: int = 1,
) -> None:
    """Draw a polyline onto an image using given line segments.

    Args:
        line_segments_arr: Array of shape (K, 2) representing the coordinates of each line segment
        image: Array of shape (M, N, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        im_h: Image height in pixels
        im_w: Image width in pixels
        thickness: thickness (in pixels) to use when rendering a polyline.
    """
    line_segments_arr = np.round(line_segments_arr).astype(np.int32)
    for i in range(line_segments_arr.shape[0] - 1):
        x1 = line_segments_arr[i][0]
        y1 = line_segments_arr[i][1]
        x2 = line_segments_arr[i + 1][0]
        y2 = line_segments_arr[i + 1][1]

        # x_in_range = (x1 >= 0) and (x2 >= 0) and (y1 >= 0) and (y2 >= 0)
        # y_in_range = (x1 < im_w) and (x2 < im_w) and (y1 < im_h) and (y2 < im_h)

        # if x_in_range and y_in_range:
        # Use anti-aliasing (AA) for curves
        image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv2.LINE_AA)


def draw_dashed_polyline(
    polyline: np.ndarray,
    image: np.ndarray,
    color: Tuple[int, int, int],
    im_h: int,
    im_w: int,
    thickness_px: int = 1,
    dash_interval_px: float = 3,
    dash_frequency: int = 2,
) -> None:
    """Draw a dashed polyline in the bird's eye view.

    If `dash_interval_px` is 1, and `dash_frequency` is 2, then we generate 1 dash every N meters,
    with equal dash-non-dash spacing.

    Image is passed by reference.
    Ignoring residual at ends, since assume lanes quite long.

    Args:
        polyline: array of shape (K, 2) representing the coordinates of each line segment.
        image: array of shape (M, N, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        im_h: Image height in pixels
        im_w: Image width in pixels
        thickness: thickness (in pixels) to use when rendering a polyline.
        dash_interval_px: length of one dash, in pixels.
        dash_frequency: for each dash_interval_m, we will discretize the length into n sections.
            1 of n sections will contain a marked dash, and the other (n-1) spaces will be empty (non-marked).
    """
    interp_polyline, num_waypts = polyline_utils.interp_polyline_by_fixed_waypt_interval(polyline, dash_interval_px)
    for i in range(num_waypts - 1):

        # every other segment is a gap
        # (first the next dash interval is a line, and then the dash interval is empty, ...)
        if (i % dash_frequency) != 0:
            continue

        dashed_segment_arr = interp_polyline[i : i + 2]
        draw_polyline_cv2(dashed_segment_arr, image, color, im_h, im_w, thickness_px)
