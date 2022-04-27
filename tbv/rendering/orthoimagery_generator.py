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
Utilities to aggregate and render sensor data in the BEV, from a single log, as the data streams in.
Sensor data includes ring camera images and potentially also LiDAR sweeps.
"""

import collections
import copy
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

import av2.rendering.color as color_utils
import av2.utils.io as io_utils
from av2.rendering.color import GREEN_HEX, RED_HEX
from av2.structures.sweep import Sweep
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.geometry.camera.pinhole_camera import PinholeCamera

# cv2.ocl.setUseOpenCL(False)

try:
    import tbv.utils.histogram_matching as histogram_matching_utils
except:
    pass
import tbv.utils.image_to_ground_correspondence as correspondence_utils
import tbv.utils.mseg_interface as mseg_interface
import tbv.synthetic_generation.map_perturbation as map_perturbation_engine
import tbv.utils.cv2_img_utils as cv2_img_utils
import tbv.utils.triangle_grid_utils as triangle_grid_utils
from tbv.rendering.bev_sensor_utils import make_bev_img_from_sensor_data
from tbv.rendering_config import BevRenderingConfig
from tbv.synthetic_generation.map_perturbation import SyntheticChangeType

from tbv.utils.seamseg import SEAMSEG_PALETTE

from tbv.eval_timestamps import EVAL_TIMESTAMPS

# from luminance_correction import run_lam_algorithm

# aggregate sensor data from at most 10 sweeps at a time, in a ring buffer
RING_BUFFER_LEN = 10

# accumulate at least 10 sweeps before we start rendering data, to avoid undesirable sparsity
N_SWEEPS_BEFORE_RENDERING = 10

# AV has to have moved at least 5 meters to render next frame (hyperparameter)
EGOMOTION_DIST_THRESH_M = 5

MINIMAL_MOTION_THRESH_M = 0.5

ORDERED_RING_CAMERA_LIST = [cam_enum.value for cam_enum in RingCameras]


logger = logging.getLogger(__name__)


def render_orthoimagery_for_logs_raytracing(
    dataloader: AV2SensorDataLoader,
    label_maps_dir: Path,
    log_id: str,
    config: BevRenderingConfig,
) -> None:
    """Render orthoimagery by finding dense correspondences between image pixels and the ground surface mesh.

    No need to accumulate over different timestamps bc of high px density.
    Just get 7 closest images to each LiDAR timestamps

    Dont use every image -> worthless.
    We use collections.deque() as a ring buffer, to keep around sensor data from a fixed number of past frames.
    Outer loop is over LiDAR sweeps, not over images. This allows us to do frame-centric rendering (use 7
    ring camera images closest to this LiDAR sweep to update the ring buffer.)
    However, we do not use any LiDAR data in this function.

    Args:
        dataloader: data loader to access TbV log data.
        label_maps_dir: directory root for where semantic segmentation label maps are saved on disk.
        log_id: string representing unique identifier for TbV log/scenario to render.
        config: specification of rendering parameters for BEV data.
    """
    log_map_dirpath = dataloader.get_log_map_dirpath(log_id=log_id)
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

    pinhole_camera_dict = {
        camera_name: dataloader.get_log_pinhole_camera(log_id=log_id, cam_name=camera_name)
        for camera_name in ORDERED_RING_CAMERA_LIST
    }

    # if car has moved significantly, re-render
    egocenter_last_rendering = np.array([0, 0])

    frustum_ring_buffer_city_pts_dict = {
        camera_name: collections.deque(maxlen=RING_BUFFER_LEN) for camera_name in ORDERED_RING_CAMERA_LIST
    }
    frustum_ring_buffer_rgb_vals_dict = {
        camera_name: collections.deque(maxlen=RING_BUFFER_LEN) for camera_name in ORDERED_RING_CAMERA_LIST
    }

    lidar_fpaths = dataloader.get_ordered_log_lidar_fpaths(log_id=log_id)
    lidar_fpaths.sort()
    num_sweeps_raytraced = 0
    for i, lidar_fpath in enumerate(lidar_fpaths):

        logger.info(f"On sweep {i}/{len(lidar_fpaths)}")
        lidar_timestamp_ns = int(Path(lidar_fpath).stem)
        print(f"On {lidar_timestamp_ns}")

        l_city_SE3_egovehicle = dataloader.get_city_SE3_ego(log_id, lidar_timestamp_ns)
        ego_center = l_city_SE3_egovehicle.translation.squeeze()[:2]
        egomotion_since_last_rendering = np.linalg.norm(ego_center - egocenter_last_rendering)
        # logger.info(f"Egovehicle moved {egomotion_since_last_rendering:.2f} m since last rendering")

        is_eval_timestamp = lidar_timestamp_ns in EVAL_TIMESTAMPS[log_id]
        sufficient_sweeps = num_sweeps_raytraced >= N_SWEEPS_BEFORE_RENDERING
        insufficient_egomotion = egomotion_since_last_rendering <= MINIMAL_MOTION_THRESH_M

        if not is_eval_timestamp and sufficient_sweeps:
            # only render from the official eval list (at train time the user could modify this
            # parameter to render more training examples, if desired)
            print(f"\tSkip {lidar_timestamp_ns} sufficient sweeps and not an eval timestamp.")
            continue
        elif not is_eval_timestamp and not sufficient_sweeps and insufficient_egomotion:
            # Raytracing at every single sweep is unnecessary if no motion. However,
            # it's useful in the beginning to get accumulation started.
            print(f"\tSkip {lidar_timestamp_ns} insufficient motion")
            continue

        # ensure all frustums will exist
        if any(
            [
                dataloader.get_closest_img_fpath(log_id, camera_name, lidar_timestamp_ns) is None
                for camera_name in ORDERED_RING_CAMERA_LIST
            ]
        ):
            print(f"\tSkip {lidar_timestamp_ns} corresponding image missing")
            continue

        if not config.render_vector_map_only:
            # actually do the ray-tracing!
            for j, camera_name in enumerate(ORDERED_RING_CAMERA_LIST):
                # print(f'{lidar_timestamp_ns} On {i} -> {camera_name}')

                cam_im_fpath = dataloader.get_closest_img_fpath(log_id, camera_name, lidar_timestamp_ns)
                if cam_im_fpath is None:
                    logger.info("missing corresponding camera image")
                    print(f"\tSkip {lidar_timestamp_ns}")
                    continue

                cam_timestamp = int(Path(cam_im_fpath).stem)
                if config.make_bev_semantic_img:
                    # this is actually the path to the semantic label map (use this instead of image captured by camera).
                    cam_im_fpath = f"{config.seamseg_output_dataroot}/{log_id}/seamseg_label_maps_{camera_name}/{cam_timestamp}.png"
                    
                if not Path(cam_im_fpath).exists():
                    logger.info("missing corresponding camera image")
                    print(f"\tSkip {lidar_timestamp_ns}")
                    continue

                if not config.render_vector_map_only:
                    rgb_img = imageio.imread(cam_im_fpath)
                    # rgb_img = run_lam_algorithm(rgb_img)

                if config.make_bev_semantic_img:
                    semantic_img = rgb_img
                    img_w = pinhole_camera_dict[camera_name].width_px
                    img_h = pinhole_camera_dict[camera_name].height_px
                    semantic_img = cv2.resize(semantic_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    # Two strategies: pretend label map is a 3-channel image, or instead use a RGB colormap for the classes.
                    # rgb_img = np.tile(rgb_img[:,:,np.newaxis], (1,1,3))

                    rgb_img = SEAMSEG_PALETTE[semantic_img]

                # egovehicle position at camera time
                city_SE3_egovehicle = dataloader.get_city_SE3_ego(log_id, cam_timestamp)

                # triangle vertices are provided in the egovehicle frame
                all_triangles = triangle_grid_utils.get_ground_surface_grid_triangles(
                    avm, city_SE3_egovehicle, range_m=config.max_ground_mesh_range_m
                )

                # all_triangles = triangle_grid_utils.get_flat_plane_grid_triangles(range_m=10)
                relevant_triangles, in_frustum = triangle_grid_utils.prune_triangles_to_2d_frustum(
                    triangles=all_triangles,
                    yaw=pinhole_camera_dict[camera_name].egovehicle_yaw_cam_rad,
                    fov_theta=pinhole_camera_dict[camera_name].fov_theta_rad,
                    margin=1
                )
                pruned_ratio = 1 - (in_frustum.sum() / in_frustum.size)
                # print(f'Discarded {pruned_ratio * 100}% of triangles, as they fell outside of camera frustum.')

                if config.filter_ground_with_semantics:
                    img_fname_stem = Path(cam_im_fpath).stem
                    label_map_path = mseg_interface.get_mseg_label_map_fpath_from_image_info(
                        label_maps_dir, log_id, camera_name, img_fname_stem
                    )
                    label_map = imageio.imread(label_map_path)
                    assert label_map.shape == rgb_img.shape[:2]
                else:
                    label_map = None

                frustum_city_pts, frustum_rgb_vals = correspondence_utils.get_point_rgb_correspondences_raytracing(
                    nearby_triangles=relevant_triangles,
                    label_map=label_map,
                    cam_timestamp=cam_timestamp,
                    rgb_img=rgb_img,
                    city_SE3_egovehicle=city_SE3_egovehicle,
                    pinhole_camera=pinhole_camera_dict[camera_name],
                )
                if config.filter_ground_with_map:
                    is_ground = avm.raster_ground_height_layer.get_ground_points_boolean(points_xyz=frustum_city_pts)
                    frustum_city_pts = frustum_city_pts[is_ground]
                    frustum_rgb_vals = frustum_rgb_vals[is_ground]

                frustum_ring_buffer_city_pts_dict[camera_name].append(frustum_city_pts)
                frustum_ring_buffer_rgb_vals_dict[camera_name].append(frustum_rgb_vals)

            if config.use_histogram_matching:
                # update just the last entries
                (
                    frustum_ring_buffer_city_pts_dict,
                    frustum_ring_buffer_rgb_vals_dict,
                ) = histogram_matching_utils.match_histograms_spatial_constraints(
                    frustum_ring_buffer_city_pts_dict, frustum_ring_buffer_rgb_vals_dict, ego_center, config, i
                )

        # print(f"Rerendering scene because threshold is {EGOMOTION_DIST_THRESH_M}")
        # if corresponding_city_pts is not None:
        #   print("Number of points to render with: ", corresponding_city_pts.shape)

        # if corresponding_city_pts.shape[0] < int(1e6):
        #   print(f'Skipping because only {corresponding_city_pts.shape[0]} points')
        #   continue

        num_sweeps_raytraced += 1
        # update latest, and record that we raytraced at this position.
        egocenter_last_rendering = ego_center
        if lidar_timestamp_ns not in EVAL_TIMESTAMPS[log_id]:
            # only render from the official eval list (at train time the user could modify this
            # parameter to render more training examples, if desired)
            continue

        if config.render_vector_map_only:
            # dummy, empty arrays
            accumulated_city_pts = np.zeros((0, 3))
            accumulated_rgb_vals = np.zeros((0, 3), dtype=np.uint8)
        else:
            pts_to_concat = []
            for camera_name in ORDERED_RING_CAMERA_LIST:
                pts_to_concat.extend(list(frustum_ring_buffer_city_pts_dict[camera_name]))
            accumulated_city_pts = np.vstack(pts_to_concat)

            rgb_to_concat = []
            for camera_name in ORDERED_RING_CAMERA_LIST:
                rgb_to_concat.extend(list(frustum_ring_buffer_rgb_vals_dict[camera_name]))
            accumulated_rgb_vals = np.vstack(rgb_to_concat)

        assert accumulated_rgb_vals.shape == accumulated_city_pts.shape

        render_scene(
            ego_center=ego_center,
            config=config,
            dataloader=dataloader,
            log_id=log_id,
            label_maps_dir=label_maps_dir,
            avm=avm,
            timestamp=lidar_timestamp_ns,
            all_city_pts=accumulated_city_pts,
            all_rgb_vals=accumulated_rgb_vals,
            city_pts_w_reflectance=None,
        )


def render_orthoimagery_for_logs_noraytracing(
    dataloader: AV2SensorDataLoader,
    label_maps_dir: Path,
    log_id: str,
    config: BevRenderingConfig,
) -> None:
    """Render orthoimagery without using any ray-tracing, but rather by finding sparse correspondences
    between image pixels and LiDAR points.

    Outer loop here is over images. We wait to start rendering until we have seen pixels from
    all ring camera frustums at least once.

    Args:
        dataloader: dataloader.
        label_maps_dir: directory root for where semantic segmentation label maps are saved on disk.
        log_id: string representing unique identifier for TbV log/scenario to render.
        config: specification of rendering parameters for BEV data.
    """
    log_map_dirpath = dataloader.get_log_map_dirpath(log_id=log_id)
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

    # we will continually append to this arrays as new sensor data arrives.
    all_city_pts = np.zeros((0, 3), dtype=np.float32)
    all_rgb_vals = np.zeros((0, 3), dtype=np.uint8)

    pinhole_camera_dict = {}
    im_fpath_cname_ts_tuples = []

    # for each camera frustum
    for camera_name in ORDERED_RING_CAMERA_LIST:
        logger.info(f"On camera {camera_name}")

        cam_im_fpaths = dataloader.get_ordered_log_cam_fpaths(log_id, camera_name)
        pinhole_camera_dict[camera_name] = dataloader.get_log_pinhole_camera(log_id=log_id, cam_name=camera_name)

        for cam_im_fpath in cam_im_fpaths:
            cam_timestamp = Path(cam_im_fpath).stem.split("_")[-1]
            cam_timestamp = int(cam_timestamp)

            im_fpath_cname_ts_tuples += [(cam_im_fpath, camera_name, cam_timestamp)]
        print(f"Added {len(cam_im_fpaths)} from {camera_name}, now {len(im_fpath_cname_ts_tuples)} tuples")

    # if car has moved significantly, re-render
    egocenter_last_rendering = np.array([0, 0])
    EGOMOTION_DIST_THRESH_M = 5

    frustums_seen = set()

    # sort all images now by timestamp
    im_fpath_cname_ts_tuples = sorted(im_fpath_cname_ts_tuples, key=lambda tup: tup[2])  # sort by cam timestamp

    if not config.render_reflectance_only or config.render_vector_map_only:
        city_pts_w_reflectance = None
    else:
        # load all of the LiDAR points once, in advance (not too many of them anyway)
        city_pts_w_reflectance = load_localized_log_reflectance_values(
            dataloader=dataloader, log_id=log_id, label_maps_dir=label_maps_dir, config=config, avm=avm
        )

    for i, (im_fpath, camera_name, cam_timestamp) in enumerate(im_fpath_cname_ts_tuples):

        frustums_seen.add(camera_name)
        logger.info(f"\tOn {i+1}/{len(im_fpath_cname_ts_tuples)}")
        img_fname_stem = Path(im_fpath).stem

        # load PLY file path, e.g. '315978406032859416.feather'
        lidar_fpath = dataloader.get_closest_lidar_fpath(log_id, cam_timestamp_ns=cam_timestamp)
        if lidar_fpath is None:
            logger.info("missing ply")
            continue
        lidar_pts = io_utils.read_lidar_sweep(lidar_fpath, attrib_spec="xyz")

        # egovehicle position at camera time
        city_SE3_egovehicle = dataloader.get_city_SE3_ego(log_id, cam_timestamp)
        if city_SE3_egovehicle is None:
            logger.info("missing camera pose")
            continue

        lidar_timestamp_ns = Path(lidar_fpath).stem
        lidar_timestamp_ns = int(lidar_timestamp_ns)

        l_city_SE3_egovehicle = dataloader.get_city_SE3_ego(log_id=log_id, timestamp_ns=lidar_timestamp_ns)
        if l_city_SE3_egovehicle is None:
            logger.info("missing LiDAR pose")
            continue

        if not config.render_reflectance_only and not config.render_vector_map_only:
            rgb_img = imageio.imread(im_fpath)
            if config.filter_ground_with_semantics:
                label_map_path = mseg_interface.get_mseg_label_map_fpath_from_image_info(
                    label_maps_dir, log_id, camera_name, img_fname_stem
                )
                label_map = imageio.imread(label_map_path)
                assert label_map.shape == rgb_img.shape[:2]
            else:
                label_map = None

            city_pts, rgb_vals = correspondence_utils.get_point_rgb_correspondences_lidar(
                avm=avm,
                config=config,
                label_map=label_map,
                log_id=log_id,
                dataloader=dataloader,
                cam_timestamp=cam_timestamp,
                lidar_timestamp_ns=lidar_timestamp_ns,
                lidar_pts=lidar_pts,
                camera_name=camera_name,
                rgb_img=rgb_img,
                city_SE3_egovehicle=city_SE3_egovehicle,
            )

            all_city_pts = np.vstack([all_city_pts, city_pts])
            all_rgb_vals = np.vstack([all_rgb_vals, rgb_vals])
        else:
            all_city_pts = None
            all_rgb_vals = None

        # have all frustums been seen once yet?
        if len(frustums_seen) < len(ORDERED_RING_CAMERA_LIST):
            print(f"Only {len(frustums_seen)} of {len(ORDERED_RING_CAMERA_LIST)} frustums seen.")
            continue

        # Alternative: could instead filter based on the sparsity of the un-interpolated image
        # # not much will be visible with less than 3 million points
        # if all_city_pts.shape[0] < int(3e6):
        #   continue

        ego_center = l_city_SE3_egovehicle.translation.squeeze()[:2]

        egomotion_since_last_rendering = np.linalg.norm(ego_center - egocenter_last_rendering)
        # logger.info(f"Egovehicle moved {egomotion_since_last_rendering:.2f} m since last rendering")
        if egomotion_since_last_rendering > EGOMOTION_DIST_THRESH_M:
            print(f"Rerendering scene because threshold is {EGOMOTION_DIST_THRESH_M}")
            if all_city_pts is not None:
                print("Number of points to render with: ", all_city_pts.shape)
            render_scene(
                ego_center=ego_center,
                config=config,
                dataloader=dataloader,
                log_id=log_id,
                label_maps_dir=label_maps_dir,
                avm=avm,
                timestamp=cam_timestamp,
                all_city_pts=all_city_pts,
                all_rgb_vals=all_rgb_vals,
                city_pts_w_reflectance=copy.deepcopy(city_pts_w_reflectance),
            )
            # update latest
            egocenter_last_rendering = ego_center


def render_scene(
    ego_center: np.ndarray,
    config: BevRenderingConfig,
    dataloader: AV2SensorDataLoader,
    log_id: str,
    label_maps_dir: Path,
    avm: ArgoverseStaticMap,
    timestamp: int,
    all_city_pts: np.ndarray,
    all_rgb_vals: np.ndarray,
    city_pts_w_reflectance: Optional[np.ndarray],
) -> None:
    """Given a 3d and color specification of a scene, generate BEV images and maps that describe it.

    These maps may be synthetic perturbations of the true map.

    Args:
        ego_center: array of shape (2,) representing (x,y) coordinates of AV/ego-vehicle in world frame.
        config: specification of rendering parameters for BEV data.
        dataloader: dataloader.
        log_id: string representing unique identifier for TbV log/scenario to render.
        label_maps_dir: directory root for where semantic segmentation label maps are saved on disk.
        avm: local map for TbV log/scenario.
        timestamp: nanosecond timestamp for ???
        all_city_pts: array of shape (N,3) representing 3d points, either intersection points where camera rays
            hit the ground surface, or coordinates of LiDAR returns classified as belonging to the ground.
        all_rgb_vals: array of shape (N,3) with values in the range [0,255]
        city_pts_w_reflectance: array of shape
    """
    # use only (x,y)

    dirname = f"_medianfilter{config.use_median_filter}"
    dirname += f"_maxrange{config.max_ground_mesh_range_m}"
    dirname += f"_filter_ground_w_semantics{config.filter_ground_with_semantics}"
    dirname += f"_filter_ground_w_map{config.filter_ground_with_map}"
    dirname += f"_res{config.resolution_m_per_px}"
    dirname += f"_prune_egocenter_{config.range_m}m"

    x, y = ego_center[:2]
    logger.info(
        f"Centered in city at (x={x*config.resolution_m_per_px:.2f}" + f",y={y*config.resolution_m_per_px:.2f})"
    )

    if not config.render_vector_map_only and not config.render_reflectance_only:

        if config.make_bev_semantic_img:
            # render semantic image as RGB BEV
            make_bev_img_from_sensor_data(
                timestamp,
                dirname,
                config,
                log_id,
                copy.deepcopy(all_city_pts),
                copy.deepcopy(all_rgb_vals),
                copy.deepcopy(ego_center),
                method=config.semantic_img_interp_type,
                use_interp=True,
                modality="semantics",
            )
        else:
            # render sensor image as RGB BEV
            # no longer using ('nearest',True) because this leads to strange artifacts along border
            make_bev_img_from_sensor_data(
                timestamp,
                dirname,
                config,
                log_id,
                copy.deepcopy(all_city_pts),
                copy.deepcopy(all_rgb_vals),
                copy.deepcopy(ego_center),
                method=config.sensor_img_interp_type,
                use_interp=True,
                modality="rgb",
            )

    if config.make_bev_semantic_img:
        # no need to render the vector map again, or perturbed versions of it
        return

    if config.render_reflectance_only or (config.render_reflectance and not config.render_vector_map_only):
        reflectance_vals = city_pts_w_reflectance[:, 3]
        grayscale_intensity_vals = equalize_distribution(reflectance_vals)
        reflectance_rgb_vals = np.tile(grayscale_intensity_vals, (3, 1)).T

        for (interp_method, use_interp) in [("linear", True), ("nearest", True), ("none", False)]:
            make_bev_img_from_sensor_data(
                timestamp,
                dirname,
                config,
                log_id=log_id,
                city_pts=copy.deepcopy(city_pts_w_reflectance[:, :3]),
                rgb_vals=copy.deepcopy(reflectance_rgb_vals),
                ego_center=copy.deepcopy(ego_center),
                method=interp_method,  # 'none', 'linear'
                use_interp=use_interp,
                modality="reflectance",
            )
    if config.render_reflectance_only:
        return

    # dont generate these for the test set
    # currently, only one change at a time
    if config.jitter_vector_map:
        for augment_type in SyntheticChangeType:
            try:
                map_perturbation_engine.render_perturbed_bev(
                    avm=avm,
                    timestamp=timestamp,
                    dirname=dirname,
                    log_id=log_id,
                    config=config,
                    ego_center=copy.deepcopy(ego_center),
                    change_type=augment_type,
                )
            except Exception as e:
                logging.exception(f"Synthetic change {augment_type} failed")

    # now render the un-modified example
    map_perturbation_engine.render_perturbed_bev(
        avm=avm,
        timestamp=timestamp,
        dirname=dirname,
        log_id=log_id,
        config=config,
        ego_center=copy.deepcopy(ego_center),
        change_type="no_change",
    )


def equalize_distribution(reflectance: np.ndarray) -> np.ndarray:
    """Add one to reflectance to map 0 values to 0 under logarithm

    Args:
        reflectance: array of shape (N,) with values in [0,255]

    Returns:
        log_reflectance: array of shape (N,) with values in [0,255] representing log of input distribution.
    """
    log_reflectance = np.log(reflectance + 1)

    # valid_idxs = np.logical_not(np.logical_or(np.isnan(log_reflectance),np.isinf(log_reflectance)))
    # valid_idxs = np.logical_and(0 < reflectance, reflectance < 5)
    # log_reflectance = log_reflectance[valid_idxs]
    # log_reflectance = reflectance[valid_idxs]

    # log_reflectance = reflectance # alternatively, use log reflectance
    log_reflectance = normalize_array(log_reflectance, max_val=255)
    log_reflectance = np.round(log_reflectance).astype(np.uint8)
    return log_reflectance  # , valid_idxs


def normalize_array(arr: np.ndarray, max_val: float = 255) -> np.ndarray:
    """
    Args:
        arr
        max_val:

    Returns:
        arr: array containing values normalized to [ , ] range
    """
    arr -= np.amin(arr)
    arr /= np.amax(arr)
    arr *= 255
    return arr


def load_localized_log_reflectance_values(
    dataloader: AV2SensorDataLoader,
    log_id: str,
    label_maps_dir: Path,
    config: BevRenderingConfig,
    avm: ArgoverseStaticMap,
) -> np.ndarray:
    """

    Args:
        dataloader:
        log_id: string representing unique identifier for TbV log/scenario to render.
        dataset_dir
        label_maps_dir: directory root for where semantic segmentation label maps are saved on disk.
        config: specification of rendering parameters for BEV data.
        avm: local map for TbV log/scenario.

    Returns:
        all_city_pts_w_refl: array of shape (N,4), representing city point coordinates w/ reflectance values
    """
    all_city_pts_w_refl = np.zeros((0, 4))

    lidar_fpaths = dataloader.get_ordered_log_lidar_fpaths(log_id=log_id)
    lidar_timestamps_ns = [int(p.stem) for p in lidar_fpaths]
    for lidar_timestamp_ns in lidar_timestamps_ns:

        city_SE3_egovehicle = dataloader.get_city_SE3_ego(log_id=log_id, timestamp_ns=lidar_timestamp_ns)
        if city_SE3_egovehicle is None:
            logger.info("Missing pose")
            continue

        lidar_fpath = Path(dataset_dir) / log_id / "lidar" / f"{lidar_timestamp_ns}.feather"
        # load with associated LiDAR intensity
        sweep_ego = Sweep.from_feather(lidar_fpath)
        ego_xyz = sweep_ego.xyz
        ego_reflectance = sweep_ego.intensity.reshape(-1, 1)
        city_xyz = city_SE3_egovehicle.transform_point_cloud(ego_xyz)

        # logic doesn't support doing both right now
        assert not (config.filter_ground_with_semantics and config.filter_ground_with_map)

        if config.filter_ground_with_semantics:
            is_ground = correspondence_utils.filter_to_ground_projected_pixels(
                lidar_pts=ego_xyz,
                loader=dataloader,
                log_id=log_id,
                lidar_timestamp_ns=lidar_timestamp_ns,
                label_maps_dir=label_maps_dir,
            )
        elif config.filter_ground_with_map:
            # use the map for the classification
            is_ground = avm.raster_ground_height_layer.get_ground_points_boolean(points_xyz=city_xyz)

        ground_city_pts = np.hstack([city_xyz[is_ground], ego_reflectance[is_ground]])
        all_city_pts_w_refl = np.vstack([all_city_pts_w_refl, ground_city_pts])

    return all_city_pts_w_refl


def render_virtual_ground_mesh_img(
    relevant_triangles: triangle_grid_utils.TRIANGLES_TYPE,
    pinhole_camera: PinholeCamera,
    camera_name: str,
    visualize: bool = False,
) -> None:
    """
    Args:
        relevant_triangles:
        pinhole_camera or dataloader
        camera_name:
        visualize:
    """
    pinhole_camera = dataloader.get_log_pinhole_camera(log_id=log_id, cam_name=camera_name)

    img = np.zeros((pinhole_camera.height_px, pinhole_camera.width_px, 3), dtype=np.uint8)

    NUM_RANGE_BINS = 20
    # repeat green to red colormap every 50 m.
    colors_arr = color_utils.create_colormap(colorlist=[RED_HEX, GREEN_HEX], n_colors=NUM_RANGE_BINS)

    for tri in relevant_triangles:
        uv, points_cam, valid_pts_bool = pinhole_camera.project_ego_to_img(points_ego=np.array(tri), remove_nan=False)
        if valid_pts_bool.sum() < 3:
            continue

        points_cam = points_cam[valid_pts_bool]
        pt_ranges = np.linalg.norm(points_cam[:, :3], axis=1)
        range_m = np.mean(pt_ranges)
        rgb_bin = np.round(range_m).astype(np.int32)
        # account for moving past 100 meters, loop around again
        rgb_bin = rgb_bin % NUM_RANGE_BINS
        uv_color = (255 * colors_arr[rgb_bin]).astype(np.int32).reshape(-1, 3)
        uv_color_bgr = np.fliplr(uv_color).squeeze()
        uv_color_bgr = tuple([int(x) for x in uv_color_bgr])
        img = cv2_img_utils.draw_polygon_cv2(uv, img, uv_color_bgr)

    if visualize:
        plt.imshow(img[:, :, ::-1])
        plt.show()


def test_render_virtual_ground_mesh_img() -> None:
    """ """
    camera_name = "ring_front_center"
    log_id = "allison-arkansas-wdc-new-bollards"

    relevant_triangles = triangle_grid_utils.get_flat_plane_grid_triangles(range_m=20)
    dataloader = None
    render_virtual_ground_mesh_img(relevant_triangles, dataloader, camera_name)


def execute_orthoimagery_job(
    dataloader: AV2SensorDataLoader, label_maps_dir: Path, log_id: str, config: BevRenderingConfig
) -> None:
    """Render the orthoimagery for a single log, either using ray-tracing for dense ground-to-image correspondence,
    or instead use image-LiDAR projection for sparse ground-to-image correspondence.

    Args:
        dataloader: dataloader.
        label_maps_dir: directory root for where semantic segmentation label maps are saved on disk.
        log_id: string representing unique identifier for TbV log/scenario to render.
        config: specification of rendering parameters for BEV data (holds all experiment parameters).
    """
    logger.info(config)

    start = time.time()
    if config.projection_method == "ray_tracing":
        render_orthoimagery_for_logs_raytracing(
            dataloader=dataloader, label_maps_dir=label_maps_dir, log_id=log_id, config=config
        )

    elif config.projection_method == "lidar_projection":
        render_orthoimagery_for_logs_noraytracing(
            dataloader=dataloader, label_maps_dir=label_maps_dir, log_id=log_id, config=config
        )
    else:
        raise RuntimeError("Unknown lidar-pixel correspondence method")

    end = time.time()
    duration = end - start
    logger.info(f"Generated orthoimagery for logs in: {duration} sec.")
