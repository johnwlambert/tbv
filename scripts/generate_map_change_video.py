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

"""Generate a mp4 video from visualizations of map changes (side-by-side map and sensor data)."""

from pathlib import Path
from typing import Final

import click
import imageio
import numpy as np
from rich.progress import track

import av2.rendering.video as video_utils
from av2.utils.typing import NDArrayByte


def generate_map_change_video(visualization_dir: Path, video_save_fpath: Path, fps: int = 10) -> None:
    """Generate an mp4 video from saved single frames.

    Each frame represents a sensor image and a map image (side-by-side).

    Args:
        visualization_dir: Path to where single frame visualizations are stored.
        video_save_fpath: Path to where mp4 will be stored.
        fps: does not correspond to original ring camera fps, since we only dump visualizations for subsampled frames.
    """
    img_fpaths = sorted(list(visualization_dir.glob("*.jpg")))

    video_list = []
    for img_fpath in track(img_fpaths, description="Concatenating frames for mp4..."):
        img_rgb = imageio.imread(img_fpath)
        video_list.append(img_rgb)
    
    video: NDArrayByte = np.stack(video_list).astype(np.uint8)
    video_utils.write_video(
        video=video,
        dst=video_save_fpath,
        fps=fps,
        preset="medium",
    )

@click.command(help="Generate a mp4 video from visualizations of map changes (side-by-side map and sensor data).")
@click.option(
    "--visualization_dir",
    help="Path to where single frame visualizations are stored.",
    default="/Users/johnlambert/Downloads/tbv-staging/triplets_2022_04_03_egoview_teaser_figure_downsampled",
    type=click.Path(exists=True),
)
@click.option(
    "--video_save_fpath",
    help="Path to where mp4 will be stored.",
    default="map_change_video.mp4",
    type=str,
)
# TODO: add flag to allow clustering by scene chronology
def run_generate_map_change_video(visualization_dir: str, video_save_fpath: str) -> None:
    """Click entry point for map change video generation."""
    generate_map_change_video(visualization_dir=Path(visualization_dir), video_save_fpath=Path(video_save_fpath))


if __name__ == "__main__":
    run_generate_map_change_video()