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
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import argoverse.utils.centerline_utils as centerline_utils
import argoverse.utils.interpolate as interp_utils
from argoverse.map_representation.map_api_v2 import LaneSegment
from argoverse.utils.cv2_plotting_utils import draw_polygon_cv2
from argoverse.utils.mesh_grid import get_mesh_grid_as_point_cloud
from argoverse.utils.polyline_density import get_polyline_length
from argoverse.utils.se2 import SE2

import tbv.rendering.crosswalk_renderer as crosswalk_renderer
from tbv.rendering.map_rendering_classes import LocalVectorMap


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


def transform_city_pts_to_img_pts(image_SE2_city: SE2, polyline: np.ndarray, resolution: float) -> np.ndarray:
    """
    Args:
        image_SE2_city
        polyline
        resolution

    Returns:
        img_polyline
    """
    img_polyline = image_SE2_city.transform_point_cloud(polyline)
    img_polyline /= resolution
    return img_polyline


def bev_render_window(
    lvm: LocalVectorMap,
    save_fpath: str,
    log_id: str,
    window_center: np.ndarray,
    resolution: float,
    dilation: float,
    line_width: float = 6,
) -> np.ndarray:
    """
    Args:
        lvm:
        save_fpath:
        log_id: string representing unique identifier for TbV log/scenario.
        window_center: Numpy array of shape 1x2
        resolution: in meters/px
        dilation: X meters in each direction around median
        line_width:

    Returns:
        bev_img: image w/ BGR colors imprinted. Up on image will also be North,
          since we flipped vertically (bc of positive y-axis mirroring)
    """
    xcenter, ycenter = window_center.astype(np.int32).squeeze()

    n_px = int(dilation * (1 / resolution))
    grid_h = n_px * 2
    grid_w = n_px * 2
    bev_img = np.ones((grid_h + 1, grid_w + 1, 3), dtype=np.uint8) * BLACK_INTENSITY
    img_h, img_w, _ = bev_img.shape

    # still operating in meters now, before shift to image space and scale
    # ymax is `ycenter + dilation` and xmax is `xcenter + dilation`
    ymin = ycenter - dilation
    xmin = xcenter - dilation

    image_SE2_city = SE2(rotation=np.eye(2), translation=np.array([-xmin, -ymin]))

    city_SE2_image = image_SE2_city.inverse()
    img_grid_pts = get_mesh_grid_as_point_cloud(min_x=0, max_x=grid_w, min_y=0, max_y=grid_h)

    img_grid_pts *= resolution

    # must account for different resolution (scale down)
    city_grid_pts = city_SE2_image.transform_point_cloud(img_grid_pts)
    inside_da = lvm.avm.get_raster_layer_points_boolean(city_grid_pts, layer_name="driveable_area")
    da_city_pts = city_grid_pts[inside_da]
    da_img_pts = image_SE2_city.transform_point_cloud(da_city_pts)

    # account back for different resolution (scale up)
    da_img_pts /= resolution
    da_img_pts = np.round(da_img_pts).astype(np.int32)
    bev_img[da_img_pts[:, 1], da_img_pts[:, 0]] = np.ones(3) * DARK_GRAY_INTENSITY

    # draw the lane polygons first
    for _, vls in lvm.nearby_lane_segment_dict.items():
        img_bndry = transform_city_pts_to_img_pts(image_SE2_city, vls.polygon_boundary[:, :2], resolution)
        draw_polygon_cv2(img_bndry, bev_img, color=GRAY_BGR)

    for _, vls in lvm.nearby_lane_segment_dict.items():
        if vls.render_r_bound:
            bev_img = render_lane_boundary(bev_img, vls, "right", resolution, image_SE2_city, line_width)
        if vls.render_l_bound:
            bev_img = render_lane_boundary(bev_img, vls, "left", resolution, image_SE2_city, line_width)

    STOPLINE_WIDTH = 0.5  # meters
    stopline_width_px = int(math.ceil(STOPLINE_WIDTH / resolution))
    for stopline in lvm.stoplines:
        img_stopline = transform_city_pts_to_img_pts(image_SE2_city, stopline, resolution)
        draw_polyline_cv2(img_stopline, bev_img, WHITE_BGR, img_h, img_w, thickness=stopline_width_px)

    # draw the crosswalks last, otherwise looks weird with stoplines on top of crosswalks
    for lpc in lvm.ped_crossing_edges:
        # render local ped crossings (lpc's)
        (edge1, edge2) = lpc.get_edges()
        img_edge1 = transform_city_pts_to_img_pts(image_SE2_city, edge1, resolution)
        img_edge2 = transform_city_pts_to_img_pts(image_SE2_city, edge2, resolution)
        # draw_polyline_cv2(img_edge1, bev_img, xwalk_color, img_h, img_w, thickness=line_width)
        crosswalk_renderer.render_rectangular_crosswalk_bev(bev_img, img_edge1.astype(np.float32), img_edge2.astype(np.float32), resolution)

    bev_img = np.flipud(bev_img)
    cv2.imwrite(save_fpath, bev_img)
    return bev_img


def render_lane_boundary(
    bev_img: np.ndarray, vls: LaneSegment, side: str, resolution: float, image_SE2_city: SE2, line_width: float
) -> np.ndarray:
    """Draw left or right lane boundary (only one of the two is rendered in this subroutine).

    Regarding lane marking patterns:
    Double lines are to be understood from the inside out, e.g. DASHED_SOLID means that the dashed line is adjacent
    to the lane carrying the property and the solid line is adjacent to the neighbor lane.

    Args:
        bev_img: array of shape (H,W,3) representing RGB raster canvas to render objects onto.
        vls: vector lane segment object
        side: string representing the side of the lane segment to paint, i.e. should be "left" or "right"
        resolution: 
        image_SE2_city: SE(2) object that maps city coordinates to image coordinates.
        line_width: width (in pixels) to use when rendering each polyline onto the BEV image.

    Returns:
        bev_img: array of shape (H,W,3) representing RGB raster canvas after objects have been rendered onto it.
    """

    img_h, img_w, _ = bev_img.shape

    if side == "right":
        polyline = vls.right_lane_boundary.xyz
        bnd_type = vls.right_mark_type
    elif side == "left":
        polyline = vls.left_lane_boundary.xyz
        bnd_type = vls.left_mark_type
    else:
        raise RuntimeError("Invalid `side` argument in `render_lane_boundary()`.")

    N_INTERP_PTS = 100
    # interpolation needs to happen before rounded to integer coordinates
    polyline = interp_utils.interp_arc(t=N_INTERP_PTS, px=polyline[:, 0], py=polyline[:, 1])
    img_polyline = transform_city_pts_to_img_pts(image_SE2_city, polyline, resolution)

    # every 1 meter for now
    dash_interval_m = 1  # meter
    dash_interval_px = dash_interval_m / resolution

    if "WHITE" in bnd_type:
        bnd_color = WHITE_BGR
    elif "YELLOW" in bnd_type:
        # bnd_color = YELLOW_BGR
        bnd_color = TRAFFIC_YELLOW1_BGR
    else:
        bnd_color = RED_BGR

    if ("DOUBLE" in bnd_type) or ("SOLID_DASH" in bnd_type) or ("DASH_SOLID" in bnd_type):
        left, right = get_double_polylines(img_polyline, width_scaling_factor=4)  # in pixels

    if bnd_type in ["SOLID_WHITE", "SOLID_YELLOW", "NONE"]:
        draw_polyline_cv2(img_polyline, bev_img, bnd_color, img_h, img_w, thickness=line_width)

    elif bnd_type in ["DOUBLE_DASH_YELLOW", "DOUBLE_DASH_WHITE"]:
        draw_dashed_polyline(
            left, bev_img, bnd_color, img_h, img_w, thickness=line_width * 2, dash_interval=dash_interval_px
        )
        draw_dashed_polyline(
            right, bev_img, bnd_color, img_h, img_w, thickness=line_width * 2, dash_interval=dash_interval_px
        )

    elif bnd_type in ["DOUBLE_SOLID_YELLOW", "DOUBLE_SOLID_WHITE"]:
        draw_polyline_cv2(left, bev_img, bnd_color, img_h, img_w, thickness=line_width * 2)
        draw_polyline_cv2(right, bev_img, bnd_color, img_h, img_w, thickness=line_width * 2)

    elif bnd_type in ["DASHED_WHITE", "DASHED_YELLOW"]:
        draw_dashed_polyline(
            img_polyline, bev_img, bnd_color, img_h, img_w, thickness=line_width, dash_interval=dash_interval_px
        )

    elif (bnd_type == "SOLID_DASH_YELLOW" and side == "right") or (bnd_type == "DASH_SOLID_YELLOW" and side == "left"):
        draw_polyline_cv2(left, bev_img, bnd_color, img_h, img_w, thickness=line_width * 2)
        draw_dashed_polyline(
            right, bev_img, bnd_color, img_h, img_w, thickness=line_width * 2, dash_interval=dash_interval_px
        )

    elif (bnd_type == "SOLID_DASH_YELLOW" and side == "left") or (bnd_type == "DASH_SOLID_YELLOW" and side == "right"):
        draw_dashed_polyline(
            left, bev_img, bnd_color, img_h, img_w, thickness=line_width * 2, dash_interval=dash_interval_px
        )
        draw_polyline_cv2(right, bev_img, bnd_color, img_h, img_w, thickness=line_width * 2)

    else:
        print(f"Unknown bnd type {bnd_type}")
        logging.exception(f"Unknown bnd type {bnd_type}")

    return bev_img


def get_double_polylines(polyline: np.ndarray, width_scaling_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Treat any polyline as a centerline, and extend a narrow strip on both sides

    Args:
        polyline: array of shape (N,2) representing a polyline.
        width_scaling_factor: 

    Returns:
        left: array of shape (?,2) representing left polyline.
        right: array of shape (?,2) representing right polyline.
    """
    double_line_polygon = centerline_utils.centerline_to_polygon(centerline=polyline, width_scaling_factor=width_scaling_factor)
    num_pts = double_line_polygon.shape[0]
    assert num_pts % 2 == 1
    # split index -- polygon from right boundary, left boundary, then close it w/ 0th vertex of right
    # we swap left and right since our polygon is generated about a boundary, not a centerline
    k = num_pts // 2
    left = double_line_polygon[:k]
    right = double_line_polygon[k:-1]  # throw away the last point, since it is just a repeat

    return left, right


def draw_polyline_cv2(
    line_segments_arr: np.ndarray,
    image: np.ndarray,
    color: Tuple[int, int, int],
    im_h: int,
    im_w: int,
    thickness: float = 1,
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
        image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness, lineType=cv2.LINE_AA)


def interp_polyline_by_fixed_wypt_interval(polyline: np.ndarray, wypt_interval: float) -> Tuple[np.ndarray, int]:
    """

    Args:
        polyline
        wypt_interval: space interval between waypoints, in meters.

    Returns:
        interp_polyline
        num_waypts
    """
    # get the total length in meters of the line segment
    len_m = get_polyline_length(polyline)

    # count number of waypoints to get the desired length
    # add one for the extra endpoint
    num_waypts = math.floor(len_m / wypt_interval) + 1
    interp_polyline = interp_utils.interp_arc(t=num_waypts, px=polyline[:, 0], py=polyline[:, 1])
    return interp_polyline, num_waypts


def draw_dashed_polyline(
    polyline: np.ndarray,
    image: np.ndarray,
    color: Tuple[int, int, int],
    im_h: int,
    im_w: int,
    thickness: float = 1,
    dash_interval: float = 3,
) -> None:
    """
    Generate 1 dash every N meters, with equal dash-non-dash spacing.
    Image is passed by reference.
    Ignoring residual at ends, since assume lanes quite long.

    Args:
        polyline: array of shape (K, 2) representing the coordinates of each line segment.
        image: array of shape (M, N, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        im_h: Image height in pixels
        im_w: Image width in pixels
        thickness:
        dash_interval:
    """
    interp_polyline, num_waypts = interp_polyline_by_fixed_wypt_interval(polyline, dash_interval)
    for i in range(num_waypts - 1):

        # every other segment is a gap
        if i % 2 == 1:
            continue

        dashed_segment_arr = interp_polyline[i : i + 2]
        draw_polyline_cv2(dashed_segment_arr, image, color, im_h, im_w, thickness)

