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

from collections import defaultdict
import copy
import csv
import glob
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import av2.utils.io as io_utils
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.map import EgoViewMapRenderer
from torch.utils.data import Dataset
from torch import Tensor

import tbv.evaluation.eval_map_change_detection as eval_map_change_detection
import tbv.utils.vis_utils as vis_utils
from tbv.rendering_config import BevRenderingConfig, EgoviewRenderingConfig, SensorViewpoint
from tbv.utils.cv2_img_utils import hstack_imgs

# Represents info about a single train/val/test example
# Represents (sensor_img_fpath, map_img_fpath, labelmap_fpath, label_idx, is_match, log_id, timestamp)
TbvExampleMetadata = Tuple[str, str, str, int, int, str, int]

# Represents (sensor_img, map_img, labelmap, label_idx, is_match, log_id, timestamp)
TbvExampleData = Tuple[Tensor, Tensor, Tensor, int, int, str, int]


CLASSNAME_TO_CLASSIDX: Dict[str, int] = {
    "no_change": 0,
    "delete_crosswalk": 1,
    "insert_crosswalk": 2,
    "change_lane_marking_color": 3,
    "delete_lane_marking": 4,
    "add_bike_lane": 5,
    "change_lane_boundary_dash_solid": 6,
}

# CLASSNAME_TO_CLASSIDX: Dict[str,int] = {
# 	'no_change': 0,
# 	'delete_crosswalk': 1,
# 	'insert_crosswalk': 2,

# 	'change_lane_marking_color': 3,
# 	'delete_lane_marking': 4,
# 	'add_bike_lane': 5,
# 	'change_lane_boundary_dash_solid': 6

# 	#'lane_geometry_change': 3 # change to more fine grained later
# }

# CLASSNAME_TO_CLASSIDX: Dict[str,int] = {
# 	'no_change': 0,
# 	'delete_crosswalk': 1,
# 	'insert_crosswalk': 2
# }


def compute_data_mean_std(data_root: str, modality: str, interp_type: str):
    """ """
    # if use sensor data only to compute the statistics
    sensor_img_fpaths = glob.glob(f"{data_root}/*_{modality}_interp{use_interp}{interp_type}*.jpg")

    rgb_vals = np.zeros((0, 1), dtype=np.uint8)
    for sensor_img_fpath in sensor_img_fpaths:
        img = imageio.imread(sensor_img_fpath)
        img = img.reshape(-1, 1)
        rgb_vals = np.vstack([rgb_vals, img])

    # if also use the map stream to compute statistics

    mean = np.mean(rgb_vals)
    std = np.std(rgb_vals)
    return mean, std


def get_split_log_ids(split: str) -> List[str]:
    """

    Args:
        split: dataset split.

    Returns:
        split_log_ids:
    """
    split_log_ids = []

    localized_test_data_fpath = "labeled_data/mcd_test_set_localization_in_space.json"
    test_data = io_utils.read_json_file(localized_test_data_fpath)
    test_log_ids = [entry["log_id"] for entry in test_data]

    all_log_ids = [Path(log_dirpath).stem for log_dirpath in glob.glob(f"{args.tbv_dataroot}/logs/*")]
    trainval_log_ids = set(all_log_ids) - set(test_log_ids)
    trainval_log_ids = list(trainval_log_ids)

    trainval_log_ids.sort()
    TRAIN_PERCENT = 90
    num_trainval_logs = len(trainval_log_ids)

    num_train_logs = int(math.floor((TRAIN_PERCENT / 100) * num_trainval_logs))
    train_log_ids = trainval_log_ids[:num_train_logs]
    val_log_ids = trainval_log_ids[num_train_logs:]
    num_val_logs = len(val_log_ids)

    logging.info(f"Of {num_trainval_logs} trainval logs, {num_train_logs} in train, {num_val_logs} in val")

    if split == "train":
        split_log_ids = train_log_ids
    elif split == "val":
        split_log_ids = val_log_ids
    else:
        raise RuntimeError("Unknown data split")

    return split_log_ids


def get_sensor_img_fpaths(args: Union[BevRenderingConfig, EgoviewRenderingConfig]) -> List[str]:
    """In BEV, projection method will either be 'lidar_projection' or 'ray_tracing'

    Args:
        args: config for dataset setup.

    Returns:
        sensor_img_fpaths: list of file paths for all sensor images in either the BEV (aggregated from images),
            or in the ego-view (raw images).
    """
    if args.viewpoint not in [SensorViewpoint.EGOVIEW, SensorViewpoint.BEV]:
        raise RuntimeError("Unknown viewpoint type" + args.viewpoint)

    if args.viewpoint == SensorViewpoint.EGOVIEW:
        # No new sensor images are rendered in the ego-view, so instead we look at rendered map images,
        # and find each corresponding camera image.
        sensor_img_fpaths = []
        # folder hierarchy resembles:
        #     {rendered_datasets_dir}/ _depthocclusionreasoningTrue / no_change / *.jpg
        rendered_img_fpaths = Path(args.rendered_dataset_dir).glob("*/*/*.jpg")

        # for every rendered image
        for rendered_img_fpath in rendered_img_fpaths:
            log_id, ts, camera_name = get_logid_ts_camera_from_egoview_rendered_path(rendered_img_fpath)
            rgb_img_fpath = f"{args.tbv_dataroot}/{log_id}/sensors/cameras/{camera_name}/{ts}.jpg"
            sensor_img_fpaths += [rgb_img_fpath]

        sensor_img_fpaths = list(set(sensor_img_fpaths))
        sensor_img_fpaths.sort()
        return sensor_img_fpaths

    elif args.viewpoint == SensorViewpoint.BEV:
        # should be one level below (the only dataset below)
        rendered_data_root = glob.glob(f"{args.rendered_dataset_dir}/*")[0]

        if args.sensor_img_interp_type in ["nearest", "linear"]:
            use_interp = True
        elif args.sensor_img_interp_type == "None":
            use_interp = False
        else:
            raise RuntimeError("Unknown interpolation type")

        sensor_img_fpaths = glob.glob(
            f"{rendered_data_root}/*_{args.sensor_modality}_interp{use_interp}{args.sensor_img_interp_type}*{args.projection_method}*.jpg"  # noqa
        )
        sensor_img_fpaths.sort()
        return sensor_img_fpaths


def get_logid_ts_camera_from_egoview_rendered_path(rendered_img_fpath: Path) -> Tuple[str, int, str]:
    """Given a map rendering in the ego-view, parse the log ID, timestamp, and camera name from the file name.

    Args:
        rendered_img_fpath: file path to a map rendering in the ego-view, e.g.
            no_change/LOgutcHPKdh7rA0PLSGWhSqGyiShi3WI__Winter_2021_ring_front_center_315973480999927214_vectormap.jpg

    Returns:
        log_id: unique ID of TbV vehicle log.
        ts: nanosecond integer timestamp.
        camera_name: name of camera, for which map viewpoint corresponds to.
    """
    fname_stem = rendered_img_fpath.stem

    j = fname_stem.find("_ring")
    log_id = fname_stem[:j]

    if "_315" not in fname_stem:
        raise RuntimeError("Invalid file name format.")

    k = fname_stem[j + 1 :].find("_315")
    camera_name = fname_stem[j + 1 : j + 1 + k]  # skip the underscore at beginning, and at the end.

    m = fname_stem[j + 1 + k + 1 :].find("_")
    ts = int(fname_stem[j + 1 + k + 1 : j + 1 + k + 1 + m])

    return log_id, ts, camera_name


def get_logid_ts_camera_from_egoview_sensor_path(img_fpath: str) -> Tuple[str, int, str]:
    """From image path in log, get metadata

    Args:
        img_fpath: path to a camera image, present within a raw vehicle log, e.g.
        '.../WY0cVNmhg7LtAs5Eny78Csltv2tbjdsd__Winter_2021/sensors/cameras/ring_front_center/315970050899927215.jpg'

    Returns:
        log_id: unique ID of TbV vehicle log.
        ts: nanosecond integer timestamp.
        camera_name: name of camera.
    """
    log_id = Path(img_fpath).parent.parent.parent.parent.stem
    camera_name = Path(img_fpath).parent.stem
    ts = int(Path(img_fpath).stem)
    return log_id, ts, camera_name


def get_logid_ts_from_bev_sensor_path(sensor_img_fpath: str, modality: str) -> Tuple[str, int]:
    """Derive log identifier and nanosecond timestamp from file path to an image, representing sensor data in the BEV.

    Args:
        sensor_img_fpath:
        modality:

    Returns:
        log_id: unique identifier for TbV log/scenario.
        timestamp_ns: integer representing nanosecond timestamp.
    """
    stem = Path(sensor_img_fpath).stem
    k = stem.find(f"_{modality}")
    log_id_ts = stem[:k]

    k = log_id_ts.find("_315")
    log_id = log_id_ts[:k]

    timestamp_ns = log_id_ts[k:]
    timestamp_ns = int(timestamp_ns.replace("_", ""))
    return log_id, timestamp_ns


def make_trainval_CE_dataset(
    split: str, args: Union[BevRenderingConfig, EgoviewRenderingConfig]
) -> List[TbvExampleMetadata]:
    """Gather metadata for examples from train or val split, to be trained with cross-entropy supervision.

    Args:
        split: dataset split.
        args: config for dataset setup.

    Returns:
        data_list:
    """
    rendered_data_root = glob.glob(f"{args.rendered_dataset_dir}/*")[0]

    if args.viewpoint not in [SensorViewpoint.EGOVIEW, SensorViewpoint.BEV]:
        raise ValueError(f"Unknown sensor viewpoint: {args.viewpoint}")

    sensor_img_fpaths = get_sensor_img_fpaths(args)
    logging.info(f"Found {len(sensor_img_fpaths)} unique sensor images")

    split_log_ids = get_split_log_ids(split)
    data_list = []
    log_ids_on_disk = set()

    split_sensor_img_fpaths = set()

    for sensor_img_fpath in sensor_img_fpaths:

        if args.viewpoint == SensorViewpoint.EGOVIEW:
            log_id, ts, camera_name = get_logid_ts_camera_from_egoview_sensor_path(sensor_img_fpath)
            labelmap_fpath = sensor_img_fpath.replace(f"/{camera_name}/", f"/seamseg_label_maps_{camera_name}/")
            labelmap_fpath = labelmap_fpath.replace(".jpg", ".png")

        elif args.viewpoint == SensorViewpoint.BEV:
            log_id, ts = get_logid_ts_from_bev_sensor_path(sensor_img_fpath, args.sensor_modality)

            labelmap_fname = Path(sensor_img_fpath).stem + ".png"
            labelmap_fname = labelmap_fname.replace("_rgb_", "_semantics_")
            labelmap_fname = labelmap_fname.replace(args.sensor_img_interp_type, args.semantic_img_interp_type)

            labelmap_fpath = f"{Path(sensor_img_fpath).parent}/semantics/{labelmap_fname}"
            if not Path(labelmap_fpath).exists():
                continue

        log_ids_on_disk.add(log_id)

        if args.viewpoint == SensorViewpoint.EGOVIEW:
            img_fname = f"{log_id}_{camera_name}_{ts}_vectormap.jpg"
        elif args.viewpoint == SensorViewpoint.BEV:
            img_fname = f"{log_id}_{ts}_vectormap.jpg"

        if log_id not in split_log_ids:
            continue
        split_sensor_img_fpaths.add(sensor_img_fpath)

        if args.use_multiple_negatives_per_sensor_img:
            # form 2-tuples for each type of synthetic change that was successful
            for change_type, label_idx in CLASSNAME_TO_CLASSIDX.items():
                map_img_fpath = f"{rendered_data_root}/{change_type}/{img_fname}"
                if not Path(map_img_fpath).exists():
                    # logging.info(f"Map img missing for {log_id}")
                    continue

                is_match = change_type == "no_change"
                data_list += [(sensor_img_fpath, map_img_fpath, labelmap_fpath, label_idx, is_match, log_id, ts)]

                if len(data_list) % 1000 == 0:
                    logging.info(f"Data list has {len(data_list)} {split} 2-tuple examples...")
        else:
            # add one positive
            map_img_fpath = f"{rendered_data_root}/no_change/{img_fname}"
            if Path(map_img_fpath).exists():
                is_match = True
                label_idx = CLASSNAME_TO_CLASSIDX["no_change"]
                data_list += [(sensor_img_fpath, map_img_fpath, labelmap_fpath, label_idx, is_match, log_id, ts)]

            # create only one negative example per sensor image (randomly chosen)
            change_types_available = []
            for change_type, label_idx in CLASSNAME_TO_CLASSIDX.items():
                map_img_fpath = f"{rendered_data_root}/{change_type}/{img_fname}"
                if Path(map_img_fpath).exists():
                    change_types_available += [change_type]
            if "no_change" in change_types_available:
                nc_idx = change_types_available.index("no_change")
                del change_types_available[nc_idx]

            if len(change_types_available) == 0:
                continue
            change_type = np.random.choice(change_types_available)
            label_idx = CLASSNAME_TO_CLASSIDX[change_type]
            map_img_fpath = f"{rendered_data_root}/{change_type}/{img_fname}"

            is_match = change_type == "no_change"
            data_list += [(sensor_img_fpath, map_img_fpath, labelmap_fpath, label_idx, is_match, log_id, ts)]

    logging.info(f"Found {len(split_sensor_img_fpaths)} sensor images for split {split}")
    print(f"Found {len(split_sensor_img_fpaths)} sensor images for split {split}")
    return data_list


def make_test_CE_dataset(
    split: str,
    args: Union[BevRenderingConfig, EgoviewRenderingConfig],
    filter_eval_by_visibility: bool,
    eval_categories: List[str],
    save_visualizations: bool = True,
) -> List[TbvExampleMetadata]:
    """Gather metadata for examples from the test split.

    Args:
        split:
        args: config for dataset setup.
        filter_eval_by_visibility: whether to evaluate only on visible nearby portions of the scene,
           or to evaluate on *all* nearby portions of the scene.
           (only useful for the "test" split.)
        eval_categories: categories to consider when selecting frames to evaluate on.
        save visualizations: whether to save side-by-side visualizations of data examples,
            i.e. an image with horizontally stacked (sensor image, map image, blended combination).

    Returns:
        data_list: list of 7-tuples, each representing
            (sensor_img_fpath, map_img_fpath, labelmap_fpath, label_idx, is_match, log_id, timestamp)
    """
    # get subdir, e.g. rendered_egoview_2022_03_29 ->
    #                  rendered_egoview_2022_03_29/_depthocclusionreasoningTrue
    rendered_data_root = glob.glob(f"{args.rendered_dataset_dir}/*")[0]

    data_list = []
    sensor_img_fpaths = get_sensor_img_fpaths(args)

    logid_to_sensorfpath_dict = defaultdict(list)

    # first, get a list of sensor file paths for each log
    for sensor_img_fpath in sensor_img_fpaths:

        if args.viewpoint == SensorViewpoint.BEV:
            log_id, ts = get_logid_ts_from_bev_sensor_path(sensor_img_fpath, args.sensor_modality)
        elif args.viewpoint == SensorViewpoint.EGOVIEW:
            log_id, ts, camera_name = get_logid_ts_camera_from_egoview_sensor_path(sensor_img_fpath)

        logid_to_sensorfpath_dict[log_id] += [sensor_img_fpath]

    # get proximity-based change event information
    logid_to_mc_events_dict = eval_map_change_detection.get_test_set_event_info_bev(data_root=Path(args.tbv_dataroot))

    # we always need the data loader, in order to fetch log poses.
    loader = AV2SensorDataLoader(data_dir=Path(args.tbv_dataroot), labels_dir=Path(args.tbv_dataroot))

    log_missing_labelmaps_dict = defaultdict(list)
    log_missing_mapimgs_dict = defaultdict(list)

    gt_counts_per_frame = defaultdict(int)
    sensor_log_ids = set()

    # Must go through one log at a time (because cannot keep all maps in memory simultaneously)
    for log_idx, log_id in enumerate(logid_to_mc_events_dict.keys()):
        sensor_img_fpaths = logid_to_sensorfpath_dict[log_id]

        if len(sensor_img_fpaths) == 0:
            print(f"========>> No images found for {log_id}! =======>>")
            continue

        # now, classify each frame individually
        if args.viewpoint == SensorViewpoint.EGOVIEW and filter_eval_by_visibility:
            # map is only needed for egoview projection
            log_map_dirpath = loader.get_log_map_dirpath(log_id=log_id)
            log_avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

        for idx, sensor_img_fpath in enumerate(sensor_img_fpaths):

            if idx % 1000 == 0:
                print(f"Log {log_idx}: Image {idx}/{len(sensor_img_fpaths)} of {log_id}")
            if args.viewpoint == SensorViewpoint.BEV:
                found_log_id, ts = get_logid_ts_from_bev_sensor_path(sensor_img_fpath, args.sensor_modality)
                assert found_log_id == log_id

                labelmap_fname = Path(sensor_img_fpath).stem + ".png"
                labelmap_fname = labelmap_fname.replace("_rgb_", "_semantics_")
                labelmap_fname = labelmap_fname.replace(args.sensor_img_interp_type, args.semantic_img_interp_type)

                labelmap_fpath = f"{Path(sensor_img_fpath).parent}/semantics/{labelmap_fname}"
                if not Path(labelmap_fpath).exists():
                    print(f"\t{idx} -> {log_id}: {labelmap_fpath} semantic label map missing on disk")
                    log_missing_labelmaps_dict[log_id] += [labelmap_fpath]
                    continue

            elif args.viewpoint == SensorViewpoint.EGOVIEW:
                found_log_id, ts, camera_name = get_logid_ts_camera_from_egoview_sensor_path(sensor_img_fpath)
                assert found_log_id == log_id

                labelmap_fpath = sensor_img_fpath.replace(f"/{camera_name}/", f"/seamseg_label_maps_{camera_name}/")
                labelmap_fpath = labelmap_fpath.replace(".jpg", ".png")

                if not Path(labelmap_fpath).exists():
                    log_missing_labelmaps_dict[log_id] += [labelmap_fpath]

            sensor_log_ids.add(log_id)

            if args.viewpoint == SensorViewpoint.EGOVIEW:
                img_fname = f"{log_id}_{camera_name}_{ts}_vectormap"

            elif args.viewpoint == SensorViewpoint.BEV:
                img_fname = f"{log_id}_{ts}_vectormap"

            # load the egovehicle's pose at this particular timestamp.
            # The distance from egovehicle to map entity will be computed.
            city_SE3_egovehicle = loader.get_city_SE3_ego(log_id=log_id, timestamp_ns=ts)

            if args.viewpoint == SensorViewpoint.EGOVIEW and filter_eval_by_visibility:
                cam_timestamp = ts
                pinhole_camera = loader.get_log_pinhole_camera(log_id=log_id, cam_name=camera_name)

            map_img_fpath = f"{rendered_data_root}/no_change/{img_fname}.jpg"
            if not Path(map_img_fpath).exists():
                print(f"Map img missing for {log_id}")
                log_missing_mapimgs_dict[log_id] += [map_img_fpath]
                continue

            mc_events = logid_to_mc_events_dict[log_id]

            if (args.viewpoint == SensorViewpoint.BEV) or (
                args.viewpoint == SensorViewpoint.EGOVIEW and not filter_eval_by_visibility
            ):
                in_range_vals = [
                    mc_event.check_if_in_range(
                        log_id=log_id, city_SE3_egovehicle=city_SE3_egovehicle, range_thresh_m=20.0
                    )
                    for mc_event in mc_events
                ]
            elif args.viewpoint == SensorViewpoint.EGOVIEW and filter_eval_by_visibility:
                in_range_vals = [
                    mc_event.check_if_in_range_egoview(
                        log_id=log_id,
                        city_SE3_egovehicle=city_SE3_egovehicle,
                        range_thresh_m=20.0,
                        pinhole_cam=pinhole_camera,
                    )
                    for mc_event in mc_events
                ]
            else:
                raise RuntimeError("Unknown parameter configuration in make_test_CE_dataset()")

            if any(in_range_vals):
                present_classes = set([e.change_type for e, irv in zip(mc_events, in_range_vals) if irv])

                gt_counts_per_frame[tuple(list(present_classes))] += 1

                ALL_CATEGORIES = ["change_lane_marking_color", "insert_crosswalk", "delete_crosswalk"]
                assert present_classes.issubset(ALL_CATEGORIES)
                if not present_classes.issubset(set(eval_categories)):
                    print("Skip: Only evaluating", eval_categories, " but got ", present_classes, sensor_img_fpath)
                    continue
            else:
                gt_counts_per_frame["no_change"] += 1

            if any(in_range_vals):
                mc_idx = np.array(in_range_vals).astype(int).argmax()
                # print('Found change in range...')
                in_range_event = mc_events[mc_idx]
                is_match = False
                label_idx = CLASSNAME_TO_CLASSIDX[in_range_event.change_type]
            else:
                is_match = True
                label_idx = CLASSNAME_TO_CLASSIDX["no_change"]
            data_list += [(sensor_img_fpath, map_img_fpath, labelmap_fpath, label_idx, is_match, log_id, ts)]

            # print(f'Example {len(data_list) - 1}', sensor_img_fpath)

            if len(data_list) % 1000 == 0:
                print(f"Data list has {len(data_list)} {split} 2-tuple examples...")

            if save_visualizations:
                vis_utils.save_egoview_sensor_map_semantics_triplet(
                    is_match,
                    img_fname,
                    sensor_img_fpath,
                    map_img_fpath,
                    labelmap_fpath,
                    save_dir=Path("./triplets_2022_04_03_egoview_teaser_figure_downsampled"),
                    render_text=True,
                )

    print(f"Missing Labelmap Stats for {args.viewpoint}:")
    for log_id, missing_labelmap_fpaths in log_missing_labelmaps_dict.items():
        print(f"\tLog {log_id}: Missing {len(missing_labelmap_fpaths)} labelmap files.")

    print(f"Missing Map Image Stats for {args.viewpoint}:")
    for log_id, missing_mapimg_fpaths in log_missing_mapimgs_dict.items():
        print(f"\tLog {log_id}: Missing {len(missing_mapimg_fpaths)} map image files.")

    change_log_ids = set(list(logid_to_mc_events_dict.keys()))
    print("Missing: ", change_log_ids - sensor_log_ids)

    print("Ground Truth Counts: ", gt_counts_per_frame)
    return data_list


def make_trainval_triplet_dataset(
    split: str,
    data_root: str,
    modality: str,
    projection_method: str,
    vis_triplet_samples: bool = False,
) -> List[Tuple[str, str, int]]:
    """Form triplets"""
    data_list = []
    sensor_img_fpaths = get_sensor_img_fpaths(args)

    raise NotImplementedError
    for sensor_img_fpath in sensor_img_fpaths:

        log_id_ts = None
        log_id, timestamp_ns = get_logid_ts_from_bev_sensor_path(sensor_img_fpath, modality)

        # get path to anchor (sensor data)
        a_path = sensor_img_fpath

        # get path to positive (true vector map)
        p_path = f"{data_root}/no_change/{log_id_ts}_vectormap.jpg"
        if not Path(p_path).exists():
            continue

        neg_classes = set(CLASSNAME_TO_CLASSIDX.keys()) - set(["no_change"])
        n_paths = []
        for neg_class in neg_classes:
            n_path = f"{data_root}/{neg_class}/{log_id_ts}_vectormap.jpg"
            if not Path(n_path).exists():
                continue

            n_paths += [n_path]

        for n_path in n_paths:
            # get path to possible negatives (synthetic v. map perturbations)
            data_list += [(a_path, p_path, n_path)]

            if vis_triplet_samples:
                # 99% of the time skip
                if np.random.rand() < 0.99:
                    continue
                save_triplet_img_figure(a_path, p_path, n_path, log_id_ts)

        if len(data_list) % 1000 == 0:
            print(f"Data list has {len(data_list)} {split} triplet examples...")

    return data_list


def make_dataset(
    split: str,
    args: Union[BevRenderingConfig, EgoviewRenderingConfig],
    form_triplets: bool,
    filter_eval_by_visibility: bool,
    eval_categories: Optional[List[str]] = None,
    save_visualizations: bool = False,
) -> List[Tuple[str, str, int]]:
    """
    train is all else for now
    val is a portion of the train data
    test is logs confirmed as pmi (or in progress)

    Args:
        split: dataset split.
        args: config for dataset setup.
        form_triplets:
        filter_eval_by_visibility: whether to evaluate only on visible nearby portions of the scene,
           or to evaluate on *all* nearby portions of the scene.
           (only useful for the "test" split.)
        eval_categories: categories to consider when selecting frames to evaluate on.
        save visualizations: whether to save side-by-side visualizations of data examples,
            i.e. an image with horizontally stacked (sensor image, map image, blended combination).
    """
    print(f"Creating data list for {split} ...")

    if split in ["train", "val"] and form_triplets == False:
        data_list = make_trainval_CE_dataset(split, args)

    elif split in ["test"] and form_triplets == False:
        data_list = make_test_CE_dataset(
            split,
            args,
            filter_eval_by_visibility=filter_eval_by_visibility,
            eval_categories=eval_categories,
            save_visualizations=save_visualizations,
        )

    elif split in ["train", "val"] and form_triplets == True:
        rendered_data_root = glob.glob(f"{args.rendered_dataset_dir}/*")[0]
        data_list = make_trainval_triplet_dataset(
            split=split,
            data_root=rendered_data_root,
            modality=args.sensor_modality,
            projection_method=args.projection_method,
        )

    else:
        raise NotImplementedError

    print("\n\n\n")
    print(f"Data list created with {len(data_list)} {split} examples.")
    print("\n\n\n")
    logging.info(f"Data list created with {len(data_list)} {split} examples.")
    return data_list


def save_triplet_img_figure(a_path: str, p_path: str, n_path: str, log_id_ts: str) -> None:
    """ """
    a_img = imageio.imread(a_path)
    p_img = imageio.imread(p_path)
    n_img = imageio.imread(n_path)

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.imshow(a_img)
    plt.title("anchor")

    plt.subplot(132)
    plt.imshow(p_img)
    plt.title("positive")

    plt.subplot(133)
    plt.imshow(n_img)
    plt.title("negative")

    fig.tight_layout(pad=1)
    plt.savefig(f"triplet_examples/{log_id_ts}.jpg", dpi=500)


# modality: 'rgb' or 'reflectance'
# interp_type: 'nearest' or 'linear' or 'none'


class McdData(Dataset):
    """Map change detection data, for training or inference."""

    def __init__(
        self,
        split: str,
        transform: Callable,
        args: Union[BevRenderingConfig, EgoviewRenderingConfig],
        filter_eval_by_visibility: bool,
        eval_categories: Optional[List[str]] = None,
        loss_type: str = "cross_entropy",
        save_visualizations: bool = False,
    ) -> None:
        """
        Suitable for CE or contrastive loss.

        Supported 2-tuples are:
                #('rgb',  'None'),
                #('rgb',  'linear'),
                #('reflectance',  'None'),
                #('reflectance',  'linear'),
                ('reflectance',  'nearest')

        Args:
            split: dataset split.
            transform: transformations to apply to the input, either to normalize/crop,
                or to also apply data augmentation.
            args: config for rendered dataset.
            filter_eval_by_visibility: whether to evaluate only on visible nearby portions of the scene,
               or to evaluate on *all* nearby portions of the scene.
               (only useful for the "test" split.)
            eval_categories: categories to consider when selecting frames to evaluate on.
            loss_type: "cross_entropy", "contrastive", or "triplet". In the paper, we use "cross_entropy" everywhere.
            save visualizations: whether to save side-by-side visualizations of data examples,
                i.e. an image with horizontally stacked (sensor image, map image, blended combination).
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unknown split: {split}")

        self.split = split
        self.transform = transform

        self.use_triplets = True if loss_type == "triplet" else False
        self.data_list = make_dataset(
            split,
            args=args,
            form_triplets=self.use_triplets,
            filter_eval_by_visibility=filter_eval_by_visibility,
            eval_categories=eval_categories,
            save_visualizations=save_visualizations,
        )

    def __len__(self):
        """Return the number examples in the dataset split."""
        return len(self.data_list)

    def __getitem__(self, index: int):
        """ """
        if self.use_triplets:
            return self.triplet_get_item(index)
        else:
            return self.twotuple_get_item(index)

    def twotuple_get_item(self, index: int) -> TbvExampleData:
        """Obtain 2-tuples for CE or Contrastive Losses"""
        sensor_img_fpath, map_img_fpath, labelmap_fpath, label_idx, is_match, log_id, timestamp = self.data_list[index]

        sensor_img = imageio.imread(sensor_img_fpath)
        map_img = imageio.imread(map_img_fpath)

        # if labelmap_fpath is None or not Path(labelmap_fpath).exists():
        #     raise RuntimeError("Semantic label map requested as input, but not found on disk.")
        # else:
        #     labelmap = imageio.imread(labelmap_fpath)
        # else:
        #     # semantic label map will NOT be employed downstream, so return dummy data.
        labelmap = copy.deepcopy(map_img)[:, :, 0]

        # hstack_img = hstack_imgs([sensor_img, map_img])
        # imageio.imwrite(f'sanity_check/{index}.jpg', hstack_img)

        if sensor_img.shape[:2] != map_img.shape[:2]:
            raise RuntimeError(f"Sensor img and map img shape mismatch: {sensor_img_fpath} {map_img_fpath}\n")
        if self.transform is not None:
            # map and sensor perturbed identically
            sensor_img, map_img, labelmap = self.transform(sensor_img, map_img, labelmap)

        if sensor_img is None:
            print("\tsensor_img was none!")
        if map_img is None:
            print("\tmap img was none!")
        if label_idx is None:
            print("\tlabel idx is none!")
        if is_match is None:
            print("\tis match is none!")

        label_idx = torch.tensor([label_idx])
        is_match = torch.tensor([int(is_match)])

        assert sensor_img is not None
        assert map_img is not None
        assert label_idx is not None
        assert is_match is not None

        # print(f'\tSensor Img: {sensor_img.shape}')
        # print(f'\tMap Img: {map_img.shape}')
        # print(f'\tLabel idx: {label_idx}')
        # print(f'\tIs match: {is_match}')

        assert isinstance(sensor_img, torch.Tensor)
        assert isinstance(map_img, torch.Tensor)
        assert isinstance(label_idx, torch.Tensor)
        assert isinstance(is_match, torch.Tensor)

        return sensor_img, map_img, labelmap, label_idx, is_match, log_id, timestamp

        # return {
        #   'sensor_img': sensor_img,
        #   'map_img': map_img,
        #   'label_idx': label_idx,
        #   'is_match': is_match
        # }

    def triplet_get_item(self, index: int):
        """Obtain 3-tuples for the Triplet Loss"""
        # anchor, positive, negative
        a_path, p_path, n_path = self.data_list[index]

        a_img = imageio.imread(a_path)
        p_img = imageio.imread(p_path)
        n_img = imageio.imread(n_path)

        h, w, _ = a_img.shape
        if not all([img.shape == (h, w, 3) for img in [a_img, p_img, n_img]]):
            raise RuntimeError(f"Sensor and map img shape mismatch {a_path}")

        if self.transform is not None:
            # feed in dummy as second argument
            dummy_label = copy.deepcopy(a_img[:, :, 0])
            a_img, _ = self.transform(a_img, dummy_label)
            p_img, _ = self.transform(p_img, dummy_label)
            n_img, _ = self.transform(n_img, dummy_label)

        return a_img, p_img, n_img
        # {
        #   'a_img': a_img,
        #   'p_img': p_img,
        #   'n_img': n_img
        # }
