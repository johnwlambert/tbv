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

import glob
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

import click
import cv2

cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras

import tbv.utils.logger_utils as logger_utils

# must come before other imports, so that things get routed to log file.
logger_utils.setup_logging()

import tbv.rendering_config as rendering_config
from tbv.splits import TRAIN, VAL, TEST
from tbv.rendering_config import BevRenderingConfig, EgoviewRenderingConfig, SensorViewpoint
from tbv.rendering.orthoimagery_generator import execute_orthoimagery_job
from tbv.rendering.egoview_vector_map_rendering_utils import execute_egoview_job
from tbv.utils.multiprocessing_utils import send_list_to_workers_with_worker_id


ORDERED_RING_CAMERA_LIST = [cam_enum.value for cam_enum in RingCameras]

# render all 1043 logs from train, real val, and real test.
LOG_IDS_TO_RENDER = list(TRAIN) + list(VAL) + list(TEST)


def render_log_imagery(
    dataloader: AV2SensorDataLoader,
    log_id: str,
    local_dataset_dir: Path,
    log_dir: Path,
    config: Union[BevRenderingConfig, EgoviewRenderingConfig],
) -> None:
    """

    Args:
        dataloader:
        log_id: unique identifier for TbV log/scenario to render.
        local_dataset_dir:
        log_dir:
        config: specification of rendering parameters for BEV or ego-view data, for experiment.
    """
    logger_utils.setup_logging()

    if config.viewpoint == SensorViewpoint.BEV:
        # Generate bird's eye view image
        execute_orthoimagery_job(
            dataloader=dataloader,
            label_maps_dir=log_dir,
            log_id=log_id,
            config=config,
        )

    elif config.viewpoint == SensorViewpoint.EGOVIEW:
        execute_egoview_job(
            dataloader=dataloader,
            log_id=log_id,
            config=config,
        )


def render_log_dataset(
    config: Union[BevRenderingConfig, EgoviewRenderingConfig],
    local_dataset_dir: Path,
    log_id: str,
    log_dir: Path,
    dataloader: AV2SensorDataLoader,
) -> None:
    """Verify that images and semantic segmentation label maps exist if required by the config.
    If the config specifies that semantic segmentation results should be completed, they will
    be executed here.

    Shared for both orthoimagery or ego-view rendering.

    Args:
        config: specification of rendering parameters for BEV or ego-view data, for experiment.
        local_dataset_dir:
        log_id: unique identifier for TbV log/scenario to render.
        log_dir:
        dataloader: dataloader object for Argoverse 2.0-style data.
    """
    for camera_name in ORDERED_RING_CAMERA_LIST:
        if not Path(f"{log_dir}/sensors/cameras/{camera_name}").exists():
            print(f"Missing one of the camera directories: {log_id} {camera_name}")
            logging.info("Missing one of the camera directories %s %s", log_id, camera_name)
            continue

    all_label_maps_exist = True
    for camera_name in ORDERED_RING_CAMERA_LIST:
        cam_label_maps_found = all(
            [
                Path(
                    f"{log_dir}/mseg-3m-480_{log_id}_{camera_name}_universal_ss/358/gray/{Path(img_fpath).stem}.png"
                ).exists()
                for img_fpath in glob.glob(f"{log_dir}/{camera_name}/*.jpg")
            ]
        )
        all_label_maps_exist = all_label_maps_exist and cam_label_maps_found

    print(f"All label maps found for {log_id}? {all_label_maps_exist}")
    logging.info(f"All label maps found for {log_id}? {all_label_maps_exist}")
    if not all_label_maps_exist and isinstance(config, BevRenderingConfig) and config.filter_ground_with_semantics:
        raise RuntimeError("Semantic label maps must be precomputed in order to filter ground w/ semantics.")

    render_log_imagery(
        dataloader=dataloader, log_id=log_id, local_dataset_dir=local_dataset_dir, log_dir=log_dir, config=config
    )


def dataset_renderer_worker(
    log_ids: List[str], start_idx: int, end_idx: int, worker_id: int, kwargs: Mapping[str, Any]
) -> None:
    """Given a list of log_ids to render, call render_log_dataset() on each of them.

    If a GPU is used (i.e. for ray-casting), we split the rendering processes evenly among the
    available GPUs.

    Args:
        log_ids: list of strings
        start_idx: integer
        end_idx: integer
        kwargs: dictionary with argument names mapped to argument values
    """
    logging.info(f"Worker {worker_id} started...")

    local_dataset_dir = kwargs["local_dataset_dir"]
    config = kwargs["config"]
    dataloader = kwargs["dataloader"]

    use_gpu = isinstance(config, BevRenderingConfig) and (
        config.recompute_segmentation or config.projection_method == "ray_tracing"
    )
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not supported on your platform.")
        # will need to use the GPU, split processes among all available gpus
        num_gpus = torch.cuda.device_count()
        gpu_id = worker_id // (config.num_processes // num_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logging.info("Creating Argoverse dataloader...")

    chunk_sz = end_idx - start_idx
    # process each image between start_idx and end_idx
    for idx in range(start_idx, end_idx):

        # sleep_duration = np.random.randint(0,60)
        # time.sleep(sleep_duration)
        if idx % 10 == 0:
            pct_completed = (idx - start_idx) / chunk_sz * 100
            logging.info(f"Completed {pct_completed:.2f}%")

        log_id = log_ids[idx]

        try:
            log_dir = Path(local_dataset_dir) / "logs" / log_id
            render_log_dataset(
                config=config,
                local_dataset_dir=local_dataset_dir,
                log_id=log_id,
                log_dir=log_dir,
                dataloader=dataloader,
            )

        except Exception as e:
            logging.exception(f"Extraction failed for {log_id}")


def render_dataset_all_logs(config: Union[BevRenderingConfig, EgoviewRenderingConfig], log_ids: List[str]) -> None:
    """Launch worker processes to render TbV data in the BEV or ego-view.

    log_id's take on the form
    '0CjqAXeTID58UXtezwdAag5zt6bpsKFp__2020-07-28-Z1F0061'

    Args:
        config: config w/ rendering parameters specified.
        log_ids: IDs of TbV vehicle logs to render.
    """
    np.random.seed(0)
    random.seed(0)

    if not (Path(config.tbv_dataroot) / "logs").exists():
        raise RuntimeError("TbV Dataset logs must be saved to {DATAROOT}/logs/")

    num_processes = config.num_processes

    # must also process the images we'll use for LiDAR later
    dataloader = AV2SensorDataLoader(
        data_dir=Path(config.tbv_dataroot) / "logs", labels_dir=Path(config.tbv_dataroot) / "logs"
    )
    for i, loaded_log_id in enumerate(dataloader.get_log_ids()):
        if i % 20 == 0:
            logging.info(f"Dataloader loaded {i}th: {loaded_log_id}")

    if num_processes == 1:
        kwargs = {"local_dataset_dir": config.tbv_dataroot, "config": config, "dataloader": dataloader}
        dataset_renderer_worker(log_ids=log_ids, start_idx=0, end_idx=len(log_ids), worker_id=0, kwargs=kwargs)
    else:
        send_list_to_workers_with_worker_id(
            num_processes=num_processes,
            list_to_split=log_ids,
            worker_func_ptr=dataset_renderer_worker,
            local_dataset_dir=config.tbv_dataroot,
            config=config,
            dataloader=dataloader,
        )


@click.command(help="Render TbV Data.")
@click.option(
    "--config_name",
    help="Provide the name of the config to use for rendering (not the path)."
    "Must be a name listed under tbv/rendering_configs/*",
    required=True,
    type=str,
)
def run_render_dataset_all_logs(config_name: str) -> None:
    """Click entry point for rendering of TbV logs."""
    logging.info(config_name)
    print(f"Rendering with {config_name}")

    # load config for experiment
    config = rendering_config.load_rendering_config(config_name)
    render_dataset_all_logs(config=config, log_ids=LOG_IDS_TO_RENDER)


if __name__ == "__main__":
    run_render_dataset_all_logs()
