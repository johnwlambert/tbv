
"""Generate list of LiDAR timestamps at which to evaluate the model (roughly every 5 meters of ego-motion)."""

from collections import defaultdict
from pathlib import Path

import click
import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from rich.progress import track

from tbv.splits import VAL, TEST


# AV has to have moved at least 5 meters to evaluate next frame
EGOMOTION_DIST_THRESH_M = 5

ORDERED_RING_CAMERA_LIST = [cam_enum.value for cam_enum in RingCameras]


def generate_eval_timestamp_list(tbv_dataroot: Path) -> None:
    """Generate list of LiDAR timestamps at which to evaluate the model.

    Args:
        tbv_dataroot: Path to local directory where the TbV logs are stored.
    """
    dataloader = AV2SensorDataLoader(data_dir=tbv_dataroot, labels_dir=tbv_dataroot)

    for split, log_ids in zip(["val", "test"], [VAL, TEST]):
        log_eval_timestamps_dict = defaultdict(list)
        for log_id in track(log_ids, description=f"Generating eval timestamps for split {split}"):

            # initialize, such that we will always get motion larger than EGOMOTION_DIST_THRESH_M at first timestamp.
            egocenter_last_rendering = np.array([float("inf"), float("inf")])

            lidar_fpaths = dataloader.get_ordered_log_lidar_fpaths(log_id=log_id)
            lidar_fpaths.sort()
            num_log_sweeps = len(lidar_fpaths)
            for i, lidar_fpath in enumerate(lidar_fpaths):
                lidar_timestamp_ns = int(Path(lidar_fpath).stem)
                # ensure all frustums will exist
                if any(
                    [
                        dataloader.get_closest_img_fpath(log_id, camera_name, lidar_timestamp_ns) is None
                        for camera_name in ORDERED_RING_CAMERA_LIST
                    ]
                ):
                    print(f"\tSkip {lidar_timestamp_ns} @ {i}/{num_log_sweeps} corresponding image missing")
                    continue

                city_SE3_egovehicle = dataloader.get_city_SE3_ego(log_id, lidar_timestamp_ns)
                ego_center = city_SE3_egovehicle.translation.squeeze()[:2]
                egomotion_since_last_rendering = np.linalg.norm(ego_center - egocenter_last_rendering)
                if egomotion_since_last_rendering <= EGOMOTION_DIST_THRESH_M:
                    continue

                log_eval_timestamps_dict[log_id].append(lidar_timestamp_ns)
                egocenter_last_rendering = ego_center

        print(f"On split {split}")
        print(log_eval_timestamps_dict)


@click.command(help="Generate list of LiDAR timestamps at which to evaluate the model.")
@click.option(
    "--tbv-dataroot",
    required=True,
    help="Path to local directory where the TbV logs are stored.",
    type=click.Path(exists=True),
)
def run_generate_eval_timestamp_list(tbv_dataroot: str) -> None:
    """Click entry point for"""
    generate_eval_timestamp_list(tbv_dataroot=Path(tbv_dataroot))


if __name__ == "__main__":
    run_generate_eval_timestamp_list()
