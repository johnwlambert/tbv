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

# must come before other imports, so that things get routed to log file.
logger_utils.setup_logging()

import tbv.rendering_config as rendering_config
from tbv.data_splits import LOG_IDS_TO_RENDER
from tbv.rendering_config import BevRenderingConfig, EgoviewRenderingConfig
from tbv.rendering.orthoimagery_generator import execute_orthoimagery_job

import cv2
import torch

cv2.ocl.setUseOpenCL(False)
from argoverse.utils.camera_stats import RING_CAMERA_LIST
from argoverse.utils.subprocess_utils import run_command
from argoverse.utils.json_utils import read_json_file

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader

from tbv.rendering.egoview_vector_map_rendering_utils import execute_egoview_job
from tbv.utils.multiprocessing_utils import send_list_to_workers_with_worker_id


def render_log_imagery(
    dataloader: AV2SensorDataLoader,
    log_id: str,
    local_dataset_dir: Path,
    log_dir: Path,
    exp_cfg: Union[BevRenderingConfig, EgoviewRenderingConfig],
) -> None:
    """

    Args:
        dataloader:
        log_id: unique identifier for TbV log/scenario to render.
        local_dataset_dir:
        log_dir:
        exp_cfg: specification of rendering parameters for BEV or ego-view data, for experiment.
    """
    logger_utils.setup_logging()

    if exp_cfg.viewpoint == "bev":
        # Generate bird's eye view image
        execute_orthoimagery_job(
            dataloader=dataloader,
            label_maps_dir=log_dir,
            log_id=log_id,
            config=exp_cfg,
        )

    elif exp_cfg.viewpoint == "egoview":
        execute_egoview_job(
            dataloader=dataloader,
            log_id=log_id,
            config=exp_cfg,
        )


def render_log_dataset(
    exp_cfg: Union[BevRenderingConfig, EgoviewRenderingConfig],
    local_dataset_dir: Path,
    log_id: str,
    log_dir: Path,
    mseg_semantic_repo_root: Path,
    dataloader: AV2SensorDataLoader,
) -> None:
    """Verify that images and semantic segmentation label maps exist if required by the config.
    If the config specifies that semantic segmentation results should be completed, they will
    be executed here.

    Shared for both orthoimagery or ego-view rendering.

    Args:
        exp_cfg: specification of rendering parameters for BEV or ego-view data, for experiment.
        local_dataset_dir:
        log_id: unique identifier for TbV log/scenario to render.
        log_dir:
        mseg_semantic_repo_root:
        dataloader: dataloader object for Argoverse 2.0-style data.
    """

    for camera_name in RING_CAMERA_LIST:
        if not Path(f"{log_dir}/sensors/cameras/{camera_name}").exists():
            print(f"Missing one of the camera directories: {log_id} {camera_name}")
            logging.info("Missing one of the camera directories %s %s", log_id, camera_name)
            continue

    all_label_maps_exist = True
    for camera_name in RING_CAMERA_LIST:
        cam_label_maps_found = all(
            [
                Path(
                    f"{slice_extraction_dir}/mseg-3m-480_{log_id}_{camera_name}_universal_ss/358/gray/{Path(img_fpath).stem}.png"
                ).exists()
                for img_fpath in glob.glob(f"{log_dir}/{camera_name}/*.jpg")
            ]
        )
        all_label_maps_exist = all_label_maps_exist and cam_label_maps_found

    print(f"All label maps found for {log_id}? {all_label_maps_exist}")
    logging.info(f"All label maps found for {log_id}? {all_label_maps_exist}")
    if not all_label_maps_exist and exp_cfg.recompute_segmentation:
        raise RuntimeError("Semantic label maps must be precomputed.")

    render_log_imagery(
        dataloader=dataloader, log_id=log_id, local_dataset_dir=local_dataset_dir, log_dir=log_dir, exp_cfg=exp_cfg
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

    local_dataset_dir = kwargs["local_dataset_dir"]
    mseg_semantic_repo_root = kwargs["mseg_semantic_repo_root"]
    exp_cfg = kwargs["exp_cfg"]

    use_gpu = (
        isinstance(exp_cfg, BevRenderingConfig)
        and (exp_cfg.recompute_segmentation or exp_cfg.projection_method == "ray_tracing")
    )
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not supported on your platform.")
        # will need to use the GPU, split processes among all available gpus
        num_gpus = torch.cuda.device_count()
        gpu_id = worker_id // (exp_cfg.num_processes // num_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logging.info("Creating Argoverse dataloader...")
    # must also process the images we'll use for LiDAR later
    dataloader = AV2SensorDataLoader(
        data_dir=Path(local_dataset_dir) / "logs", labels_dir=Path(local_dataset_dir) / "logs"
    )
    for i, loaded_log_id in enumerate(dataloader.get_log_ids()):
        if i % 20 == 0:
            logging.info(f"Dataloader loaded {i}th: {loaded_log_id}")

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
            log_dir = f"{local_dataset_dir}/logs/{log_id}"
            render_log_dataset(
                exp_cfg=exp_cfg,
                local_dataset_dir=local_dataset_dir,
                log_id=log_id,
                log_dir=log_dir,
                mseg_semantic_repo_root=mseg_semantic_repo_root,
                dataloader=dataloader,
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


def render_dataset_all_logs(
    exp_cfg: Union[BevRenderingConfig, EgoviewRenderingConfig], mseg_semantic_repo_root: Path
) -> None:
    """
    Can have GPU thread, and also orthoimagery, simultaneously without blocking.

    log_id's take on the form
    '0CjqAXeTID58UXtezwdAag5zt6bpsKFp__2020-07-28-Z1F0061'

    Args:
        exp_cfg:
        mseg_semantic_repo_root:
    """
    np.random.seed(0)
    random.seed(0)

    local_dataset_dir = exp_cfg.tbv_dataroot

    if not (Path(exp_cfg.tbv_dataroot) / "logs").exists():
        raise RuntimeError("TbV Dataset logs must be saved to {DATAROOT}/logs/")

    # check_mkdir(f"{local_dataset_dir}/logs")
    # check_mkdir(f"{local_dataset_dir}/maps")

    log_ids = LOG_IDS_TO_RENDER
    num_processes = exp_cfg.num_processes

    if num_processes == 1:
        kwargs = {
            "local_dataset_dir": local_dataset_dir,
            "mseg_semantic_repo_root": mseg_semantic_repo_root,
            "exp_cfg": exp_cfg,
        }
        dataset_renderer_worker(log_ids=log_ids, start_idx=0, end_idx=len(log_ids), worker_id=0, kwargs=kwargs)
    else:
        # if we are on a 8-gpu machine
        # assert num_processes % 8 == 0
        send_list_to_workers_with_worker_id(
            num_processes=num_processes,
            list_to_split=log_ids,
            worker_func_ptr=dataset_renderer_worker,
            local_dataset_dir=local_dataset_dir,
            mseg_semantic_repo_root=mseg_semantic_repo_root,
            exp_cfg=exp_cfg,
        )


if __name__ == "__main__":
    """Pass the name of the desired config for rendering, via the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default="",
        # required=True,
        help="provide the name of the config to use for rendering",
    )

    parser.add_argument(
        "--mseg_semantic_repo_root",
        type=str,
        default="/Users/jlambert/Downloads/mseg-semantic",
        # f"{HOME_DIR}/Documents/mseg-semantic"
        # f"{HOME_DIR}/mseg-semantic"
        #
        # required=True,
        help="provide the name of the config to use for rendering",
    )

    args = parser.parse_args()
    logging.info(args)
    print(args)

    # load config for experiment
    # test_subsampled_label_maps_exist()

    # exp_cfg_path = 'configs/train_2020_09_12_lidar_rgb_corr_config.yaml'
    # exp_cfg_path = 'configs/train_2020_09_16_lidar_rgb_corr_synthetic_config.yaml'
    # exp_cfg_path = 'configs/train_2020_10_24_lidar_rgb_corr_synthetic_config.yaml'
    # exp_cfg_path = 'configs/train_2020_10_26_lidar_rgb_corr_synthetic_config.yaml'

    # exp_cfg_path = 'configs/train_2020_10_29_lidar_rgb_corr_synthetic_config_v2.yaml'
    # exp_cfg_path = 'configs/train_2020_11_03_lidar_rgb_corr_synthetic_config_v2.yaml' # big ray-traced
    # exp_cfg_path = 'configs/train_2020_11_04_lidar_rgb_corr_synthetic_config_v2.yaml' # experimental luminance

    # exp_cfg_path = "configs/train_2020_11_04_lidar_rgb_corr_config_v1.yaml"

    # exp_cfg_path = 'configs/train_2020_11_10_lidar_rgb_corr_synthetic_config_v1.yaml' # experimental luminance
    # exp_cfg_path = 'configs/train_2020_11_14_lidar_rgb_corr_synthetic_config_paperfighistmatchsem.yaml'

    # exp_cfg_path = 'configs/train_2020_12_14_egoview_synthetic_config_v1.yaml'
    # exp_cfg_path = 'configs/train_2021_01_04_egoview_synthetic_config_v1.yaml'

    # exp_cfg_path = 'configs/train_2021_02_05_seamseg_bev_config_v1.yaml'
    # exp_cfg_path = 'configs/train_2021_02_09_seamseg_bev_debug_config_v1.yaml'

    # exp_cfg_path = 'configs/train_2021_03_13_lidar_reflectance_bev_gen_config_v1.yaml'

    # exp_cfg_path = 'configs/train_2021_04_17_egoview_synthetic_config_v1.yaml'
    # exp_cfg_path = "configs/train_2021_04_23_lidar_rgb_corr_config_v1.yaml"

    # exp_config_name = "train_2021_09_03_egoview_synthetic_config_v1.yaml"
    # exp_config_name = "render_2021_09_03_egoview_synthetic_config_t5820.yaml"
    # exp_config_name = "render_2021_09_04_bev_synthetic_config_t5820.yaml"
    # exp_config_name = "render_2021_09_03_egoview_synthetic_config_v1.yaml"
    exp_config_name = "render_2021_09_04_bev_synthetic_config_t5820.yaml"

    args.config_name = exp_config_name

    exp_config = rendering_config.load_rendering_config(args.config_name)
    render_dataset_all_logs(exp_cfg=exp_config, mseg_semantic_repo_root=Path(args.mseg_semantic_repo_root))
