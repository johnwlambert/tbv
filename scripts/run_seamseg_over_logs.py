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

"""Run the `seamseg` semantic segmentation model over TbV logs.

Note: We modify `seamseg`'s `test_panoptic.py` script to:
(1) not use DDP.
(2) accept an `out_dir` input argument via CLI.
(3) not re-run inference if the label map already exists on disk, where we expect.
"""

import glob
import logging
import os
import shutil
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional

import click
import numpy as np
import torch

from av2.datasets.sensor.constants import RingCameras

import tbv.utils.logger_utils as logger_utils

logger_utils.setup_logging()

from tbv.splits import TRAIN, VAL, TEST
from tbv.utils.subprocess_utils import run_command
from tbv.utils.multiprocessing_utils import send_list_to_workers_with_worker_id


ORDERED_RING_CAMERA_LIST = [camera_enum.value for camera_enum in list(RingCameras)]


TBV_SPLITS = {
    "train": TRAIN,
    "val": VAL,
    "test": TEST,
}


EPSILON = 1e-9


def check_rmtree(dirpath: str) -> None:
    """ """
    if Path(dirpath).exists():
        shutil.rmtree(dirpath)
    assert not Path(dirpath).exists()


def run_log_inference(
    log_id: str,
    tbv_dataroot: Path,
    seamseg_model_dirpath: Path,
    seamseg_output_dataroot: Path,
    camera_name: str = "ring_front_center",
) -> None:
    """Run seamseg over all images of a specific camera from a specific log.

    Args:
        log_id:
        tbv_dataroot: Path to local directory where the TbV logs are stored.
        seamseg_model_dirpath
        seamseg_output_dataroot: New directory where `seamseg` output (semantic label maps) will be saved.
        camera_name:
    """
    if not isinstance(tbv_dataroot, Path):
        raise ValueError("`tbv_dataroot` input arg must be a Path object.")

    if not isinstance(seamseg_output_dataroot, Path):
        raise ValueError("`seamseg_output_dataroot` input arg must be a Path object.")

    img_dir = tbv_dataroot / log_id / "sensors" / "cameras" / camera_name
    label_map_output_dir = seamseg_output_dataroot / log_id / f"seamseg_label_maps_{camera_name}"
    label_map_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = "python test_panoptic.py"
    cmd += f" --meta {seamseg_model_dirpath}/metadata.bin"
    cmd += f" --log_dir ../logs {seamseg_model_dirpath}/config.ini"
    cmd += f" {seamseg_model_dirpath}/seamseg_r50_vistas.tar {img_dir} {label_map_output_dir}"

    print(cmd)
    run_command(cmd)

    # check how many images have corresponding label maps
    # rgb_fpaths = glob.glob(f'{img_dir}/*.jpg')
    # pred_fpaths = glob.glob(f'{label_map_output_dir}/*.png')

    # rgb_fname_stems = set([ Path(fp).stem for fp in rgb_fpaths ])
    # pred_fname_stems = set([ Path(fp).stem for fp in pred_fpaths ])

    # inter = rgb_fname_stems.intersection(pred_fname_stems)
    # union = rgb_fname_stems.union(pred_fname_stems)
    # iou = len(inter) / ( len(union) + EPSILON ) * 100
    # print(f'{iou:.2f}% for {log_id}')

    # check_rendering_percent(log_id, mcd_dataset_id)


def camera_inference_worker(
    log_ids: List[str], start_idx: int, end_idx: int, worker_id: int, kwargs: Mapping[str, Any]
) -> None:
    """Set up seamseg inference calls that will be executed by the current worker.

    Args:
        log_ids: list of TbV log IDs to be processed by the current worker.
        start_idx: integer
        end_idx: integer
        kwargs: dictionary with argument names mapped to argument values
    """
    tbv_dataroot = kwargs["tbv_dataroot"]
    seamseg_model_dirpath = kwargs["seamseg_model_dirpath"]
    seamseg_output_dataroot = kwargs["seamseg_output_dataroot"]
    camera_names = kwargs["camera_names"]
    num_processes = kwargs["config"].num_processes

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not supported on your platform.")
    # will need to use the GPU, split processes among all available gpus
    num_gpus = torch.cuda.device_count()
    gpu_id = worker_id // (num_processes // num_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    chunk_sz = end_idx - start_idx
    # process each image between start_idx and end_idx
    for idx in range(start_idx, end_idx):

        # sleep_duration = np.random.randint(0,300)
        # time.sleep(sleep_duration)
        if idx % 1000 == 0:
            pct_completed = (idx - start_idx) / chunk_sz * 100
            logging.info(f"Completed {pct_completed:.2f}%")

        log_id = log_ids[idx]
        try:
            logging.info(f"Worker {worker_id}: Commencing {log_id}")
            for camera_name in camera_names:
                run_log_inference(
                    log_id=log_id,
                    tbv_dataroot=tbv_dataroot,
                    seamseg_model_dirpath=seamseg_model_dirpath,
                    seamseg_output_dataroot=seamseg_output_dataroot,
                    camera_name=camera_name,
                )
        except Exception as e:
            logging.exception(f"Worker {worker_id}: Extraction failed for {log_id}")


def run_inference_over_camera_images(
    log_ids: List[str],
    tbv_dataroot: Path,
    seamseg_model_dirpath: Path,
    seamseg_output_dataroot: Path,
    num_processes: int,
    split: str,
    infer_ring_front_center_only: bool,
) -> None:
    """Infer semantic label maps for all logs, for images from specified cameras.

    Args:
        log_ids
        tbv_dataroot: Path to local directory where the TbV logs are stored.
        seamseg_model_dirpath:
        seamseg_output_dataroot: New directory where `seamseg` output (semantic label maps) will be saved.
        num_processes: Number of separate processes to use for inference.
        split: TbV dataset split to infer semantic label maps for.
        infer_ring_front_center_only: Whether to infer semantic label maps for the front-center camera frustum only,
            or for all camera frustums.
    """
    camera_names = ["ring_front_center"] if infer_ring_front_center_only else ORDERED_RING_CAMERA_LIST

    # Note: `num_processes` is passed to worker function via a `config` object.
    # We cannot have repeated arg name, so must pass in via config obj.
    send_list_to_workers_with_worker_id(
        num_processes=num_processes,
        list_to_split=log_ids,
        worker_func_ptr=camera_inference_worker,
        tbv_dataroot=tbv_dataroot,
        seamseg_model_dirpath=seamseg_model_dirpath,
        seamseg_output_dataroot=seamseg_output_dataroot,
        camera_names=camera_names,
        config=SimpleNamespace(**{"num_processes": num_processes}),
    )


def all_label_maps_exist_locally(tbv_dataroot: Path, seamseg_output_dataroot: Path, log_id: str) -> None:
    """

    Args:
        tbv_dataroot: Path to local directory where the TbV logs are stored.
        seamseg_output_dataroot: New directory where `seamseg` output (semantic label maps) will be saved.
        log_id:

    Returns:
        Fraction of sensor images that have a corresponding label map on disk, from `seamseg` inference.
    """
    EPS = 1e-10

    # count the number of `ring_front_center` images.
    sensor_fpaths = (tbv_dataroot / log_id / "sensors" / "cameras" / "ring_front_center").glob("*.jpg")

    # count the number of rendered label maps.
    label_map_fpaths = (seamseg_output_dataroot / log_id / "seamseg_label_maps_ring_front_center").glob("*.png")

    num_sensor_imgs = len(list(sensor_fpaths))
    num_label_maps = len(list(label_map_fpaths))
    completion_fraction = num_sensor_imgs / (num_label_maps + EPS)
    print(f"Log {log_id}: {completion_fraction * 100:.2f}")
    return completion_fraction


def estimate_completion_percentage(
    log_ids: List[str], tbv_dataroot: Path, seamseg_output_dataroot: Path, num_processes: int = 10
) -> None:
    """Estimate the percentage of label maps that have been inferred thus far.

    Args:
        log_ids:
        tbv_dataroot: Path to local directory where the TbV logs are stored.
        seamseg_output_dataroot: New directory where `seamseg` output (semantic label maps) have been saved.
        num_processes: Number of separate processes to use for counting label maps.
    """
    args = [(tbv_dataroot, seamseg_output_dataroot, log_id) for log_id in log_ids]
    with Pool(num_processes) as p:
        per_log_completion_frac_list = p.starmap(all_label_maps_exist_locally, args)

    percentage_complete = np.array(per_log_completion_frac_list).mean() * 100
    print(f"percentage complete: {percentage_complete:.2f}%")


@click.command(help="Evaluate predictions on the val or test split of the TbV Dataset.")
@click.option(
    "--tbv-dataroot",
    required=True,
    help="Path to local directory where the TbV logs are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "--seamseg_model_dirpath",
    required=True,
    help="Path to directory containing 3 files: config.ini, metadata.bin, seamseg_r50_vistas.tar",
    type=click.Path(exists=True),
)
@click.option(
    "--seamseg_output_dataroot",
    required=True,
    help="New directory where `seamseg` output (semantic label maps) will be saved.",
    type=str,
)
@click.option("--num-processes", required=True, help="Number of separate processes to use for inference.", type=int)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "test"], case_sensitive=True),
    required=True,
    help="TbV dataset split to infer semantic label maps for.",
)
@click.option(
    "--infer_ring_front_center_only",
    default=True,
    help="Whether to infer semantic label maps for the front-center camera frustum only, or for all camera frustums.",
    type=bool,
)
@click.option(
    "--log-id",
    default=None,
    help="If a specific log ID is desired, it can be passed here. Otherwise, uses all discovered log_ids in the split.",
    type=str,
)
def run_seamseg_over_logs(
    tbv_dataroot: str,
    seamseg_model_dirpath: str,
    seamseg_output_dataroot: str,
    num_processes: int,
    split: str,
    infer_ring_front_center_only: bool,
    log_id: Optional[str],
) -> None:
    """Click entry point for `seamseg` semantic segmentation inference over TbV data."""

    if log_id is None:
        log_ids = TBV_SPLITS[split]
    else:
        log_ids = [log_id]

    run_inference_over_camera_images(
        log_ids=log_ids,
        tbv_dataroot=Path(tbv_dataroot),
        seamseg_model_dirpath=Path(seamseg_model_dirpath),
        seamseg_output_dataroot=Path(seamseg_output_dataroot),
        num_processes=num_processes,
        split=split,
        infer_ring_front_center_only=infer_ring_front_center_only,
    )
    """
    estimate_completion_percentage(
        log_ids=log_ids,
        tbv_dataroot=Path(tbv_dataroot),
        seamseg_output_dataroot=Path(seamseg_output_dataroot),
        num_processes=num_processes,
    )
    """


if __name__ == "__main__":
    run_seamseg_over_logs()
