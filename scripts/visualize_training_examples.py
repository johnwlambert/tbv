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
from typing import Optional

import cv2
from pathlib import Path

import click
import imageio
import matplotlib.pyplot as plt
import numpy as np

import tbv.utils.vis_utils as vis_utils
from tbv.synthetic_generation.map_perturbation import SyntheticChangeType

"""
Visualize triplets of sensor imagery, unchanged maps, and changed map examples, side-by-side.

Works for both BEV datasets and ego-view datasets. Can also use test.py for ego-view visualization.
"""
ALPHA = 0.5

BORDER_SZ = 20

# each image is resized to 1000x1000.
# To visualize @ training resolution, set WINDOW_SZ to 224.
WINDOW_SZ = 1000


def visualize_training_examples(
    perspective: str, log_id: str, data_root: Optional[Path], rendered_data_dir: Path, output_dir: Path
) -> None:
    """
    Need to figure out the multi-polygon case, since it crashes for now

        _reflectance_interpTruenearest_projmethodlidar_projection.jpg
        _rgb_interpFalseNone_projmethodlidar_projection.jpg
        _rgb_interpTruelinear_projmethodlidar_projection.jpg
        _reflectance_interpFalseNone_projmethodlidar_projection.jpg
        _reflectance_interpTruelinear_projmethodlidar_projection.jpg

    Args:
        perspective: Data perspective (either egoview or bev).
        log_id: unique log identifier.
        data_root: Path to local directory where the TbV logs are stored.
        rendered_data_dir: Path to local directory where renderings are saved.
        output_dir: Path to local directory where before/after change visualizations will be saved.
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # projmethod = 'lidar_projection'
    projmethod = "ray_tracing"
    modality_interptype_tuples = [
        # ('rgb', 'False', 'None'),
        ("rgb", "True", "linear"),
        # ('reflectance', 'False', 'None'),
        # ('reflectance', 'True', 'linear'),
        # ('reflectance', 'True', 'nearest')
    ]

    map_change_types = [sct.lower() for sct in SyntheticChangeType]
    for change_type in map_change_types:
        for (modality, use_interp, interp_type) in modality_interptype_tuples:

            if perspective == "bev":
                sensor_img_fpaths = list(
                    Path(rendered_data_dir).glob(f"*{log_id}*_{modality}_interp{use_interp}{interp_type}*.jpg")
                )
                # sensor_img_fpaths = list(Path(rendered_data_dir).glob(f"{log_id}*.jpg"))

            elif perspective == "egoview":
                sensor_img_fpaths = list(Path(data_root).glob(f"{log_id}/sensors/cameras/ring_front_center/*.jpg"))

            num_sensor_imgs = len(sensor_img_fpaths)
            step_sz = 1
            for i in range(0, num_sensor_imgs, step_sz):
                sensor_img_fpath = sensor_img_fpaths[i]
                stem = Path(sensor_img_fpath).stem

                if perspective == "bev":
                    k = stem.find(f"_{modality}")
                    log_id_ts = stem[:k]
                    k = log_id_ts.find("_315")
                    assert log_id_ts[:k] == log_id
                    timestamp = log_id_ts[k + 1 :]

                elif perspective == "egoview":
                    timestamp = Path(sensor_img_fpath).stem
                    k = stem.find("_vectormap")
                    log_id_ts = f"{log_id}_ring_front_center_{timestamp}"

                map_change_img_fpath = Path(rendered_data_dir) / change_type / f"{log_id_ts}_vectormap.jpg"
                if not Path(map_change_img_fpath).exists():
                    continue

                sensor_img = imageio.imread(sensor_img_fpath)

                map_change_img = imageio.imread(map_change_img_fpath)
                map_change_img = cv2.resize(map_change_img, (WINDOW_SZ, WINDOW_SZ))
                sensor_img = cv2.resize(sensor_img, (WINDOW_SZ, WINDOW_SZ))

                no_change_img_fpath = Path(rendered_data_dir) / "no_change" / f"{log_id_ts}_vectormap.jpg"
                if not Path(no_change_img_fpath).exists():
                    # continue
                    no_change_img = np.zeros_like(sensor_img)
                else:
                    no_change_img = imageio.imread(no_change_img_fpath)
                no_change_img = cv2.resize(no_change_img, (WINDOW_SZ, WINDOW_SZ))

                triplet_img = vis_utils.hstack_imgs_w_border(
                    [sensor_img, no_change_img, map_change_img], border_sz=BORDER_SZ
                )

                triplet_save_fpath = Path(output_dir) / f"{change_type}_{Path(sensor_img_fpath).stem}.jpg"
                imageio.imwrite(triplet_save_fpath, triplet_img)


@click.command(help="Generate visualizations of TbV training examples.")
@click.option(
    "--perspective",
    default=None,
    help="Data perspective (either egoview or bev).",
    type=str,
)
@click.option(
    "-l",
    "--log-id",
    default=None,
    help="unique log identifier.",
    type=str,
)
@click.option(
    "-d",
    "--data-root",
    default=None,
    help="Path to local directory where the TbV logs are stored.",
    type=str,
)
@click.option(
    "--rendered-data-dir",
    required=True,
    help="Path to local directory where renderings are saved.",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    help="Path to local directory where before/after change visualizations will be saved.",
    type=str,
)
def run_visualize_training_examples(
    perspective: str, log_id: str, data_root: str, rendered_data_dir: str, output_dir: str
) -> None:
    """Click entry point for ..."""
    if perspective == "egoview" and data_root is None:
        raise ValueError("Must provide a path to the raw sensor data, to visualize egoview examples.")

    # find log ids
    if log_id is None:
        dl = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)
        log_ids = dl.get_valid_ids()
    else:
        log_ids = [log_id]

    for log_id in log_ids:
        visualize_training_examples(
            perspective=perspective,
            log_id=log_id,
            data_root=Path(data_root) if data_root is not None else None,
            rendered_data_dir=Path(rendered_data_dir),
            output_dir=Path(output_dir),
        )


if __name__ == "__main__":
    run_visualize_training_examples()
