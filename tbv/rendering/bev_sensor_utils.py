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

import time
from pathlib import Path

import av2.utils.dense_grid_interpolation as dense_grid_interpolation
import imageio
import numpy as np

import tbv.utils.logger_utils as logger_utils
from tbv.rendering_config import BevRenderingConfig
from tbv.utils.median_filter import form_aggregate_rgb_img
from tbv.utils.se2 import SE2


logger = logger_utils.get_logger()


def prune_to_2d_bbox(pts: np.ndarray, xmin: float, ymin: float, xmax: float, ymax: float) -> np.ndarray:
    """Filter a point set to the subset that resides within a specified 2d box.

    Args:
        bev_img_pts, xmin, ymin, xmax, ymax

    Returns:
        valid_xy: logicals
    """
    x = pts[:, 0]
    y = pts[:, 1]
    valid_x = np.logical_and(xmin <= x, x < xmax)
    valid_y = np.logical_and(ymin <= y, y < ymax)
    valid_xy = np.logical_and(valid_x, valid_y)
    return valid_xy


def make_bev_img_from_sensor_data(
    timestamp: int,
    dirname: str,
    config: BevRenderingConfig,
    log_id: str,
    pts_city: np.ndarray,
    rgb_vals: np.ndarray,
    ego_center: np.ndarray,
    method: str = "linear",
    use_interp: bool = False,
    modality: str = "rgb",
) -> None:
    """Generate an orthographic texture map in a bird's eye view from panoramic perspective cameras.

    Note: The size of the bird's eye view image is a function of the resolution and range of 3d points
    to consider away from the egovehicle.

    For example, we could generate BEV ground height imagery for 20 meters in all directions around
    egovehicle. `Linear` interp shows fewer speckle artifacts than `nearest`.

    Since the positive y axes are mirrored in North-South city frame
    and image frame, a up-down flip is required before saving the image.

    City:
            (2,5) C  D (5,5)
            (2,2) A  B (5,2)

    Naive Image:
            (2,2) A  B (2,5)
            (5,2) C  D (5,5)

    TODO: rotate so that +x for vehicle (facing forward) is +x in image plane (towards right).

    Args:
        timestamp:
        dirname:
        config:
        log_id:
        pts_city: float array of shape (N,3)
        rgb_vals: uint8 array of shape (N,3)
        ego_center: Numpy array of shape (2,) with (x,y) coordinates of egovehicle's center
        method: either "linear" or "nearest".
        use_interp: whether to densely interpolate the texture map from sparse pixel values.
        modality:
    """
    if config.make_bev_semantic_img:
        img_suffix = ".png"
    else:
        # for sensor image
        img_suffix = ".jpg"

    im_fname = f"{log_id}_{timestamp}"
    im_fname += f"_{modality}"
    im_fname += f"_interp{use_interp}{method}"
    im_fname += f"_projmethod{config.projection_method}"
    im_fname += img_suffix

    save_dir = Path(config.rendered_dataset_dir) / dirname
    if config.make_bev_semantic_img:
        save_dir = save_dir / "semantics"

    save_dir.mkdir(exist_ok=True, parents=True)
    save_fpath = save_dir / im_fname
    if save_fpath.exists():
        # no need to regenerate this image.
        logger.info("BEV sensor image already exists, so skipping rendering...")
        return

    logger.info(
        f"Generating BEV image using {method} interpolation from " + f"{config.projection_method} correspondence..."
    )

    # range_m describes X meters in each direction around median
    n_px = int(config.range_m * (1 / config.resolution_m_per_px))
    grid_h = n_px * 2
    grid_w = n_px * 2
    bev_img = np.zeros((grid_h + 1, grid_w + 1, 3), dtype=np.uint8)

    # Prune after we rescale. resolution is given in m/px
    pts_city /= config.resolution_m_per_px
    ego_center /= config.resolution_m_per_px

    pts_city = np.round(pts_city)
    xcenter, ycenter = np.round(ego_center).astype(np.int32)

    ymin = ycenter - n_px
    ymax = ycenter + n_px
    xmin = xcenter - n_px
    xmax = xcenter + n_px

    # --- prune to relevant region --------------
    logicals = prune_to_2d_bbox(pts_city, xmin, ymin, xmax, ymax)
    rgb_vals = rgb_vals[logicals]
    pts_city = pts_city[logicals]
    # -------------------------------------------

    bevimg_SE2_city = SE2(rotation=np.eye(2), translation=np.array([-xmin, -ymin]))

    pts_bevimg = bevimg_SE2_city.transform_point_cloud(pts_city[:, :2])
    pts_bevimg = (pts_bevimg).astype(np.int32)

    # if use_interp:
    # 	# fill in sparse grid with pixel values, dense interp
    # 	bev_img = interp_dense_grid_from_sparse(bev_img, bev_img_pts, rgb_vals, grid_h, grid_w, method)
    # else:
    # 	method = None
    # 	logger.info(f'Sparse {rgb_vals.shape[0]} of Dense {bev_img.shape[0] * bev_img.shape[1]}')

    # 	if config.use_median_filter:
    # 		bev_img = form_aggregate_rgb_img(bev_img, bev_img_pts, rgb_vals, method='median')
    # 	elif config.use_mean_filter:
    # 		bev_img = form_aggregate_rgb_img(bev_img, bev_img_pts, rgb_vals, method='mean')
    # 	else:
    # 		bev_img = form_aggregate_rgb_img(bev_img, bev_img_pts, rgb_vals, method='naive')

    if use_interp:
        bev_img = interp_dense_grid_at_unique_locations(bev_img, pts_bevimg, rgb_vals, grid_h, grid_w, method)
    else:
        bev_img = form_aggregate_rgb_img(bev_img, pts_bevimg, rgb_vals, method="naive")

    bev_img = np.flipud(bev_img)

    MAX_SPARSITY_PERCENT_THRESH = 99
    num_empty_px = (bev_img[:, :, 0] == 0).sum()
    num_px = bev_img[:, :, 0].size
    sparsity_percent = num_empty_px / num_px * 100
    if sparsity_percent > MAX_SPARSITY_PERCENT_THRESH:
        # is_too_sparse = False # don't use for now
        logger.info(f"Sparsity exceeded limits: {sparsity_percent}% vs. {MAX_SPARSITY_PERCENT_THRESH}")
    # 	return is_too_sparse
    # else:
    # 	is_too_sparse = False

    imageio.imwrite(save_fpath, bev_img)

    # return is_too_sparse


def interp_dense_grid_at_unique_locations(
    bev_img: np.ndarray, bev_img_pts: np.ndarray, rgb_vals: np.ndarray, grid_h: int, grid_w: int, method: str
) -> np.ndarray:
    """
    too much data unless we subsample (100x redundancy)
    reduce redundancy
    first fill in a grid
    then pull out the nonzero coordinates
    then interpolate only those values

    Args:
        bev_img: array of shape (H,W) representing
        bev_img_pts: array of shape () representing
        rgb_vals: array of shape () representing
        grid_h:
        grid_w:
        method: interpolation method, either "linear" or "nearest"

    Returns:
        bev_img: array of shape () representing.
    """
    start = time.time()
    bev_img = form_aggregate_rgb_img(bev_img, bev_img_pts, rgb_vals, method="naive")

    # sum up two different copies
    # only use the indices that werent used for the first one

    y, x = np.where(bev_img[:, :, 0] != 0)
    unique_bev_img_pts = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
    last_rgb_vals = bev_img[y, x]

    bev_img = dense_grid_interpolation.interp_dense_grid_from_sparse(
        grid_img=bev_img,
        points=unique_bev_img_pts,
        values=last_rgb_vals,
        grid_h=grid_h,
        grid_w=grid_w,
        interp_method=method,
    )

    end = time.time()
    duration = end - start
    print(f"Interpolation took {duration:.2f} sec")
    return bev_img
