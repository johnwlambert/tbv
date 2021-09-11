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
Script to launch rendering jobs in parallel. Each rendering job is specific to a single
log, and so N processes handle N logs at a time (one per process).
"""

import argparse
import glob
import logging
import numpy as np
import os
import random
import shutil
import time
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Mapping, Tuple, Union

import tbv.utils.logger_utils as logger_utils
from tbv.utils.dir_utils import check_mkdir

# HOME_DIR = '/home/ubuntu'
HOME_DIR = "/home/jlambert"
# HOME_DIR = "/Users/jlambert/Downloads"

# must come before other imports, so that things get routed to log file.
logger_utils.setup_logging(HOME_DIR)

import tbv.rendering_config as rendering_config
import tbv.utils.mseg_interface as mseg_interface
from tbv.data_splits import LOG_IDS_TO_RENDER
from tbv.rendering_config import BevRenderingConfig, EgoviewRenderingConfig
from tbv.rendering.make_bev_rgb_img import execute_orthoimagery_job

import cv2

cv2.ocl.setUseOpenCL(False)
from argoverse.utils.camera_stats import RING_CAMERA_LIST
from argoverse.utils.subprocess_utils import run_command
from argoverse.utils.json_utils import read_json_file
from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader

from tbv.rendering.egoview_vector_map_rendering_utils_open3d import execute_egoview_job

# from tbv.rendering.egoview_vector_map_rendering_utils import execute_egoview_job
from tbv.utils.multiprocessing_utils import send_list_to_workers_with_worker_id


def render_log_dataset(
    exp_cfg: Union[BevRenderingConfig, EgoviewRenderingConfig],
    mcd_dataset_id: str,
    local_dataset_dir: str,
    log_id: str,
    slice_extraction_dir: str,
    mseg_semantic_repo_root: str,
    mcd_repo_root: str,
    dl: SimpleArgoverseTrackingDataLoader,
    all_maps_dir: str,
) -> None:
    """
    Could be orthoimagery or ego-view alike
    TODO: make this global hyperparam (should be an args param instead)

    Args:
        exp_cfg: specification of rendering parameters for BEV or ego-view data, for experiment.
        mcd_dataset_id:
        local_dataset_dir:
        log_id:
        slice_extraction_dir:
        mseg_semantic_repo_root:
        mcd_repo_root:
        dl: dataloader object for Argoverse-style data.
        all_maps_dir:
    """
    city_info = read_json_file(f"{slice_extraction_dir}/city_info.json")
    city_id = city_info["city_id"]

    for camera_name in RING_CAMERA_LIST:
        if not Path(f"{slice_extraction_dir}/{camera_name}").exists():
            print("Missing one of the camera directories")
            logging.info("Missing one of the camera directories")
            continue

    all_label_maps_exist = True
    for camera_name in RING_CAMERA_LIST:
        cam_label_maps_found = all(
            [
                Path(
                    f"{slice_extraction_dir}/mseg-3m-480_{log_id}_{camera_name}_universal_ss/358/gray/{Path(img_fpath).stem}.png"
                ).exists()
                for img_fpath in glob.glob(f"{slice_extraction_dir}/{camera_name}/*.jpg")
            ]
        )
        all_label_maps_exist = all_label_maps_exist and cam_label_maps_found

    print(f"All label maps found for {log_id}? {all_label_maps_exist}")
    logging.info(f"All label maps found for {log_id}? {all_label_maps_exist}")
    if not all_label_maps_exist and exp_cfg.recompute_segmentation:
        logging.info("Running semantic segmentation inference...")

        # cmd = f'bash {mcd_repo_root}/run_ground_imagery.sh {subsampled_slice_extraction_dir} {mseg_semantic_repo_root}'
        # stdout, stderr = run_command(cmd, return_output=True)
        # dump_txt_report(stdout, reports_dir, log_id, report_type_id='MSegInference')
        mseg_interface.run_semantic_segmentation(log_id, slice_extraction_dir, mseg_semantic_repo_root)

    render_log_imagery(all_maps_dir, log_id, local_dataset_dir, slice_extraction_dir, exp_cfg)


def render_log_imagery(
    all_maps_dir: str,
    log_id: str,
    local_dataset_dir: str,
    slice_extraction_dir: str,
    exp_cfg: Union[BevRenderingConfig, EgoviewRenderingConfig],
) -> None:
    """

    Args:
        all_maps_dir:
        log_id:
        local_dataset_dir:
        slice_extraction_dir:
        exp_cfg: specification of rendering parameters for BEV or ego-view data, for experiment.
    """
    logger_utils.setup_logging(HOME_DIR)

    if exp_cfg.viewpoint == "bev":
        # Generate bird's eye view image
        execute_orthoimagery_job(
            maps_storage_dir=all_maps_dir,
            data_dir=f"{local_dataset_dir}/logs",
            label_maps_dir=slice_extraction_dir,
            log_id=log_id,
            config=exp_cfg,
        )

        # move the bird's eye images to the log folder
        bev_img_fpaths = glob.glob(f"{log_id}*.jpg")
        bev_img_fpaths.extend(glob.glob(f"{log_id}*.png"))
        for bev_img_fpath in bev_img_fpaths:
            fname = Path(bev_img_fpath).name
            dst = f"{slice_extraction_dir}/{fname}"
            copyfile(bev_img_fpath, dst)

            dst = f"{local_dataset_dir}/bev_imagery/{fname}"
            copyfile(bev_img_fpath, dst)

    elif exp_cfg.viewpoint == "egoview":
        execute_egoview_job(
            maps_storage_dir=all_maps_dir,
            data_dir=f"{local_dataset_dir}/logs",
            label_maps_dir=slice_extraction_dir,
            log_id=log_id,
            config=exp_cfg,
        )


def dataset_renderer_worker(
    log_ids: List[str], start_idx: int, end_idx: int, worker_id: int, kwargs: Mapping[str, Any]
) -> None:
    """Given a list of log_ids to render call render_log_dataset on each of them.
    Args:
        log_ids: list of strings
        start_idx: integer
        end_idx: integer
        kwargs: dictionary with argument names mapped to argument values
    """
    logging.info(f"Worker {worker_id} started...")

    mcd_dataset_id = kwargs["mcd_dataset_id"]
    local_dataset_dir = kwargs["local_dataset_dir"]
    mseg_semantic_repo_root = kwargs["mseg_semantic_repo_root"]
    mcd_repo_root = kwargs["mcd_repo_root"]
    exp_cfg = kwargs["exp_cfg"]

    # all_maps_dir = f'{local_dataset_dir}/maps'
    all_maps_dir = f"{local_dataset_dir}/TbV_v1.1_vector_maps_bydir"

    if (exp_cfg.recompute_segmentation or exp_cfg.projection_method == "ray_tracing") and HOME_DIR == "/home/ubuntu":
        # will need to use the GPU
        # HARD-CODED FOR NOW!
        gpu_id = worker_id // (exp_cfg.num_processes // 8)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # only need the dataloader for reflectance generation.
    if exp_cfg.render_reflectance:
        logging.info("Creating Argoverse dataloader...")
        # must also process the images we'll use for LiDAR later
        dl = SimpleArgoverseTrackingDataLoader(
            data_dir=f"{local_dataset_dir}/logs", labels_dir=f"{local_dataset_dir}/logs"
        )
        for i, loaded_log_id in enumerate(dl.sdb.get_valid_logs()):
            if i % 20 == 0:
                logging.info(f"Dataloader loaded {i}th: {loaded_log_id}")
    else:
        dl = None

    chunk_sz = end_idx - start_idx
    # process each image between start_idx and end_idx
    for idx in range(start_idx, end_idx):

        # sleep_duration = np.random.randint(0,60)
        # time.sleep(sleep_duration)
        if idx % 10 == 0:
            pct_completed = (idx - start_idx) / chunk_sz * 100
            logging.info(f"Completed {pct_completed:.2f}%")

        log_id = log_ids[idx]

        # if exp_cfg.viewpoint == "egoview" and not
        # 	print(f'Egoview: Skip {log_id} since not all seamseg label maps exist')
        # 	continue # TODO: check if seamseg label maps exist

        try:
            slice_extraction_dir = f"{local_dataset_dir}/logs/{log_id}"
            render_log_dataset(
                exp_cfg,
                mcd_dataset_id,
                local_dataset_dir,
                log_id,
                slice_extraction_dir,
                mseg_semantic_repo_root,
                mcd_repo_root,
                dl,
                all_maps_dir,
            )
            # # Will be deleted even if exception occurs
            # if exp_cfg.delete_log_after_rendering:
            #     check_rmtree(slice_extraction_dir)

        except Exception as e:
            logging.exception(f"Extraction failed for {log_id}")


def check_rmtree(dirpath: str) -> None:
    """ """
    if Path(dirpath).exists():
        shutil.rmtree(dirpath)
    assert not Path(dirpath).exists()


def render_dataset_all_logs(exp_cfg: Union[BevRenderingConfig, EgoviewRenderingConfig]) -> None:
    """
    Can have GPU thread, and also orthoimagery, simultaneously without blocking.

    log_id's take on the form
    '0CjqAXeTID58UXtezwdAag5zt6bpsKFp__2020-07-28-Z1F0061'

    Args:
        exp_cfg:
    """
    np.random.seed(0)
    random.seed(0)

    if HOME_DIR == "/home/jlambert":
        # on t5820
        mcd_repo_root = f"{HOME_DIR}/Documents/hd-map-change-detection"
        mseg_semantic_repo_root = f"{HOME_DIR}/Documents/mseg-semantic"

    elif HOME_DIR == "/home/ubuntu":
        # on ec2
        mcd_repo_root = f"{HOME_DIR}/hd-map-change-detection"
        mseg_semantic_repo_root = f"{HOME_DIR}/mseg-semantic"

    elif HOME_DIR == "/Users/jlambert/Downloads":
        mcd_repo_root = "/Users/jlambert/Downloads/tbv"
        mseg_semantic_repo_root = "/Users/jlambert/Downloads"

    else:
        raise RuntimeError("Unknown home directory")

    mcd_dataset_id = "mcd_extraction_output_dir_q85_v12_2020_08_15_02_02_50"
    exp_cfg.mcd_dataset_id = mcd_dataset_id

    local_dataset_dir = exp_cfg.tbv_dataroot

    # local_dataset_dir = f"{HOME_DIR}/{mcd_dataset_id}"
    # check_mkdir(f"{local_dataset_dir}/logs")
    # check_mkdir(f"{local_dataset_dir}/maps")

    log_ids = LOG_IDS_TO_RENDER

    num_processes = exp_cfg.num_processes

    if num_processes == 1:
        kwargs = {
            "mcd_dataset_id": mcd_dataset_id,
            "local_dataset_dir": local_dataset_dir,
            "mseg_semantic_repo_root": mseg_semantic_repo_root,
            "mcd_repo_root": mcd_repo_root,
            "exp_cfg": exp_cfg,
        }
        dataset_renderer_worker(log_ids, start_idx=0, end_idx=len(log_ids), worker_id=0, kwargs=kwargs)
    else:
        if HOME_DIR == "/home/ubuntu":
            assert num_processes % 8 == 0
        send_list_to_workers_with_worker_id(
            num_processes=num_processes,
            list_to_split=log_ids,
            worker_func_ptr=dataset_renderer_worker,
            mcd_dataset_id=mcd_dataset_id,
            local_dataset_dir=local_dataset_dir,
            mseg_semantic_repo_root=mseg_semantic_repo_root,
            mcd_repo_root=mcd_repo_root,
            exp_cfg=exp_cfg,
        )


if __name__ == "__main__":
    """Pass the name of the desired config for rendering, via the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default="",
        required=True,
        help="provide the name of the config to use for rendering",
    )
    args = parser.parse_args()
    logging.info(args)
    print(args)

    exp_config = rendering_config.get_rendering_config(args.config_name)
    render_dataset_all_logs(exp_config)
