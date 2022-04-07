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

import copy
import logging
import os
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import av2.utils.io as io_utils
import cv2
import imageio
import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.map import EgoViewMapRenderer

import tbv.rendering.crosswalk_renderer as crosswalk_renderer
import tbv.rendering.polygon_rasterization as polygon_rasterization
import tbv.synthetic_generation.map_perturbation as map_perturbation_utils
from tbv.common.local_vector_map import LocalVectorMap
from tbv.rendering.bev_map_renderer import (
    WHITE_BGR,
    draw_polyline_cv2,
)
from tbv.rendering_config import EgoviewRenderingConfig
from tbv.synthetic_generation.map_perturbation import SyntheticChangeType


logger = logging.getLogger(__name__)


GRAY_BGR = [168, 168, 168]
DARK_GRAY_BGR = [100, 100, 100]

EGOMOTION_DIST_THRESH_M = 5

# apply uniform dashes. in reality, there is often a roughly 2:1 ratio between empty space and dashes.
DASH_INTERVAL_M = 1.0  # every 1 meter


def cam_timestamp_from_img_fpath(img_fpath: str) -> int:
    """ """
    cam_timestamp = Path(img_fpath).stem
    return int(cam_timestamp)


def lidar_timestamp_from_lidar_fpath(lidar_fpath: str) -> int:
    """ """
    lidar_timestamp = Path(lidar_fpath).stem
    return int(lidar_timestamp)


def execute_egoview_job(dataloader: AV2SensorDataLoader, log_id: str, config: EgoviewRenderingConfig) -> None:
    """Render the map from the ring front center camera's viewpoint every time the egovehicle traverses N meters.

    Note: we don't perturb the map for images from the test set. Currently we apply only one change at a time.

    Args:
        dataloader: dataloader.
        log_id: unique ID for TbV scenario/log.
        config: hyperparameters for rendering.
    """
    log_map_dirpath = dataloader.get_log_map_dirpath(log_id=log_id)
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

    dirname = f"_depthocclusionreasoning{config.use_depth_map_for_occlusion}"

    # only using ring front center for the time being
    camera_name = "ring_front_center"

    cam_im_fpaths = dataloader.get_ordered_log_cam_fpaths(log_id, camera_name)
    num_cam_imgs = len(cam_im_fpaths)

    # if car has moved significantly, re-render
    egocenter_last_rendering = np.array([0, 0])

    for i, im_fpath in enumerate(cam_im_fpaths):
        if i % 50 == 0:
            logging.info(f"\tOn file {i}/{num_cam_imgs} of camera {camera_name} of {log_id}")

        cam_timestamp = cam_timestamp_from_img_fpath(im_fpath)
        city_SE3_egovehicle = dataloader.get_city_SE3_ego(log_id, cam_timestamp)
        if city_SE3_egovehicle is None:
            logger.info("missing LiDAR pose")
            continue

        ego_center = city_SE3_egovehicle.translation.squeeze()[:2]
        egomotion_since_last_rendering = np.linalg.norm(ego_center - egocenter_last_rendering)
        # logger.info(f"Egovehicle moved {egomotion_since_last_rendering:.2f} m since last rendering")
        if egomotion_since_last_rendering <= EGOMOTION_DIST_THRESH_M:
            continue

        # # load feather lidar file path, e.g. '315978406032859416.feather'
        lidar_fpath = dataloader.get_closest_lidar_fpath(log_id, cam_timestamp)
        if lidar_fpath is None:
            # without depth map, cant do this accurately
            continue

        # update the last ego-center now
        egocenter_last_rendering = ego_center

        lidar_pts = io_utils.read_lidar_sweep(lidar_fpath, attrib_spec="xyz")
        lidar_timestamp = lidar_timestamp_from_lidar_fpath(lidar_fpath)

        # save_img_fpath = f"{save_dir}/{camera_name}_{cam_timestamp}.jpg"

        # Swap channel order as OpenCV expects it -- BGR not RGB
        # must make a copy to make memory contiguous
        img_bgr = cv2.imread(str(im_fpath))
        img_h, img_w, _ = img_bgr.shape

        depth_map = dataloader.get_depth_map_from_lidar(
            lidar_points=lidar_pts,
            cam_name=camera_name,
            log_id=log_id,
            cam_timestamp_ns=cam_timestamp,
            lidar_timestamp_ns=lidar_timestamp,
            interp_depth_map=True,
        )
        if depth_map is None:
            logger.info("Depth map creation failed, skipping this image frame...")
            continue

        pinhole_camera = dataloader.get_log_pinhole_camera(log_id=log_id, cam_name=camera_name)
        egovehicle_SE3_city = city_SE3_egovehicle.inverse()
        ego_metadata = EgoViewMapRenderer(
            depth_map=depth_map,
            city_SE3_ego=city_SE3_egovehicle,
            pinhole_cam=pinhole_camera,
            avm=avm,
        )

        if config.jitter_vector_map and not config.render_test_set_only:
            for augment_type in SyntheticChangeType:
                try:
                    render_perturbed_egoview(
                        ego_metadata=ego_metadata,
                        timestamp=cam_timestamp,
                        dirname=dirname,
                        log_id=log_id,
                        config=config,
                        ego_center=copy.deepcopy(ego_center),
                        change_type=augment_type,
                    )
                except Exception as e:
                    logging.exception(f"Synthetic change {augment_type} failed")

        # now render the un-modified example
        render_perturbed_egoview(
            ego_metadata=ego_metadata,
            timestamp=cam_timestamp,
            dirname=dirname,
            log_id=log_id,
            config=config,
            ego_center=copy.deepcopy(ego_center),
            change_type="no_change",
        )


def render_perturbed_egoview(
    ego_metadata: EgoViewMapRenderer,
    timestamp: int,
    dirname: str,
    log_id: str,
    config: EgoviewRenderingConfig,
    ego_center: np.ndarray,
    change_type: Union[SyntheticChangeType, str] = "no_change",
) -> None:
    """Synthetically manipulate a vector map, render the map in the ego-view, and save rendering to disk.

    Args:
        ego_metadata:
        timestamp:
        dirname:
        log_id:
        config: hyperparameters for rendering in the ego-view.
        ego_center: location of egovehicle in the city coordinate frame, as (x,y) in meters.
        change_type: type of synthetic change to apply, or `no_change` to preserve as intact.
    """
    save_dir = Path(config.rendered_dataset_dir) / dirname / change_type.lower()
    save_dir.mkdir(parents=True, exist_ok=True)
    im_fname = f"{log_id}_{ego_metadata.pinhole_cam.cam_name}_{timestamp}_vectormap.jpg"
    save_fpath = save_dir / im_fname
    if save_fpath.exists():
        logging.info("BEV map rendering already exists, so skipping rendering...")
        return

    lvm = map_perturbation_utils.create_and_perturb_local_vector_map(
        avm=ego_metadata.avm,
        timestamp=timestamp,
        log_id=log_id,
        config=config,  # fix later
        ego_center=ego_center,
        change_type=change_type,
    )
    if lvm is None:
        return

    start = time.time()
    img = render_egoview_with_occlusion_checks(
        ego_metadata=ego_metadata, lvm=lvm, ego_center=ego_center, range_m=config.range_m, change_type=change_type
    )
    end = time.time()
    duration = end - start
    print(f"Rendering single image took {duration:.2f} sec.")
    if img is None:
        return

    imageio.imwrite(save_fpath, img)


def render_egoview_with_occlusion_checks(
    ego_metadata: EgoViewMapRenderer,
    lvm: LocalVectorMap,
    ego_center: np.ndarray,
    range_m: float,
    change_type: str,
) -> Optional[np.ndarray]:
    """Render a map in the ego-view.

    Args:
        ego_metadata:
        lvm: local vector map.
        ego_center: location of egovehicle in the city coordinate frame, as (x,y) in meters.
        range_m: maximum range from egovehicle to consider for rendering (by l-infinity norm).
        change_type: type of synthetic change to apply, or `no_change` to preserve as intact.

    Returns:
        RGB image (H,W,3)
    """
    img_h, img_w = ego_metadata.depth_map.shape
    img_bgr = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    if change_type != "no_change":
        # there should be a changed entity
        if lvm.changed_points is None:
            logging.info("Changed entity was none, skipping...")
            return None

        if lvm.changed_points.shape[1] != 3 or lvm.changed_points.ndim != 2:
            import pdb

            pdb.set_trace()
            logging.info("Changed entity was wrong shape " + str(lvm.changed_points.shape) + ", skipping...")
            return None

        changed_points_egofr = ego_metadata.ego_SE3_city.transform_point_cloud(lvm.changed_points)
        logging.info(
            "Changed points (mean="
            + str(np.round(changed_points_egofr.mean(axis=0)))
            + f") projected into frustum {ego_metadata.pinhole_cam.cam_name}"
        )

        # try projecting changed entity --if doesnt fall into frustum, return early
        # TODO: choose camera name based on where change is
        _, _, valid_pts_bool = ego_metadata.pinhole_cam.project_ego_to_img(
            points_ego=changed_points_egofr, remove_nan=False
        )
        if valid_pts_bool.sum() == 0:
            logging.info(
                "None of the changed points (mean="
                + str(np.round(changed_points_egofr.mean(axis=0)))
                + f") projected into frustum {ego_metadata.pinhole_cam.cam_name}"
            )
            return None

    # Render the drivable area, as triangles
    DA_RANGE = 100

    # counterclockwise order
    polygon_pts_cityfr = np.vstack(
        [
            ego_center.reshape(1, 2) + np.array([DA_RANGE, -DA_RANGE]).reshape(1, 2),
            ego_center.reshape(1, 2) + np.array([DA_RANGE, DA_RANGE]).reshape(1, 2),
            ego_center.reshape(1, 2) + np.array([-DA_RANGE, DA_RANGE]).reshape(1, 2),
            ego_center.reshape(1, 2) + np.array([-DA_RANGE, -DA_RANGE]).reshape(1, 2),
        ]
    )
    img_bgr = polygon_rasterization.render_polygon_egoview(
        ego_metadata,
        img_bgr,
        polygon_pts_cityfr,
        downsample_factor=0.1,
        allow_interior_only=False,
        filter_to_driveable_area=True,
        color_bgr=DARK_GRAY_BGR,
    )

    # draw the lane polygons first
    for _, vls in lvm.nearby_lane_segment_dict.items():
        img_bgr = polygon_rasterization.render_polygon_egoview(
            ego_metadata,
            img_bgr,
            polygon_pts_cityfr=vls.polygon_boundary,
            downsample_factor=0.1,
            allow_interior_only=True,
            filter_to_driveable_area=False,
            color_bgr=GRAY_BGR,
        )

    line_width_px = 10
    for _, vls in lvm.nearby_lane_segment_dict.items():
        if vls.render_r_bound:
            img_bgr = ego_metadata.render_lane_boundary_egoview(img_bgr, vls, side="right", line_width_px=line_width_px)

        if vls.render_l_bound:
            img_bgr = ego_metadata.render_lane_boundary_egoview(img_bgr, vls, side="left", line_width_px=line_width_px)

    # STOPLINE_WIDTH = 0.5 # meters
    # stopline_width_px = int(math.ceil(STOPLINE_WIDTH/resolution))
    # for stopline in lvm.stoplines:
    #     img_stopline = transform_city_pts_to_img_pts(image_SE2_city, stopline, resolution)
    #     draw_polyline_cv2(img_stopline, bev_img, WHITE_BGR, img_h, img_w, thickness=stopline_width_px)

    # draw the crosswalks last, looks weird with stoplines on top of crosswalks
    for lpc in lvm.ped_crossing_edges:
        # render local ped crossings (lpc's)
        crosswalk_renderer.render_rectangular_crosswalk_egoview(
            ego_metadata, img_bgr, lpc.edge1[:, :2], lpc.edge2[:, :2]
        )

    return img_bgr[:, :, ::-1]
