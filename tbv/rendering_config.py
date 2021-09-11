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

from dataclasses import dataclass
from typing import Optional, Union

import hydra
from hydra.utils import instantiate

# todo: put default args at the end


@dataclass
class RenderingConfig:
    """Holds options for rendering training data in the ego-view or BEV.

    Args:
        num_processes
        tbv_dataroot: path to root TbV Dataset directory.
        viewpoint: dataset rendering perspective, either "bev" or "egoview"
    """

    # could differentiate between CPU and GPU processes, TODO: MUST BE MULTIPLE OF 8
    num_processes: int
    render_vector_map_only: bool
    render_reflectance: bool
    recompute_segmentation: bool

    dilation: float  # 20 m in all directions

    tbv_dataroot: str
    rendered_dataset_dir: str

    render_test_set_only: bool
    jitter_vector_map: bool
    max_dist_to_del_crosswalk: float
    delete_log_after_rendering: bool
    viewpoint: str


@dataclass
class EgoviewRenderingConfig(RenderingConfig):
    """
    Args:
    use_depth_map_for_occlusion:
    """

    use_depth_map_for_occlusion: bool


@dataclass
class BevRenderingConfig(RenderingConfig):
    """

    Args:
        projection_method: for BEV only, either to use 'ray_tracing' for correspondences, or 'lidar_projection'
            TODO: make it a enum
    """

    projection_method: str
    use_histogram_matching: bool
    make_bev_semantic_img: bool  # whether to render RGB or semantics in BEV
    render_reflectance_only: bool
    filter_ground_with_semantics: bool  # quite expensive
    filter_ground_with_map: bool  # using ground height
    max_ground_mesh_range_m: float  # trace rays to triangles up to 20 m away
    resolution: float  # meters/pixel
    use_median_filter: bool
    sensor_img_interp_type: str  # linear for rgb, but nearest for label map
    # linear for rgb, but nearest for label map, if not rendering semantics, optional
    semantic_img_interp_type: Optional[str] = None  


def get_bev_rendering_config(config_name: str) -> BevRenderingConfig:
    """Get experiment config for rendering maps and sensor data in a bird's eye view."""
    with hydra.initialize_config_module(config_module="tbv.configs"):
        # config is relative to the tbv module
        cfg = hydra.compose(config_name=config_name)
        config: BevRenderingConfig = instantiate(cfg.BevRenderingConfig)

    return config


def get_egoview_rendering_config(config_name: str) -> EgoviewRenderingConfig:
    """Get experiment config for rendering maps and sensor data in the ego-view."""
    with hydra.initialize_config_module(config_module="tbv.configs"):
        # config is relative to the tbv module
        cfg = hydra.compose(config_name=config_name)
        config: EgoviewRenderingConfig = instantiate(cfg.EgoviewRenderingConfig)

    return config


def get_rendering_config(config_name: str) -> Union[EgoviewRenderingConfig, BevRenderingConfig]:
    """Load from disk the parameters for rendering sensor+map data (from a yaml file)."""
    try:
        config = get_egoview_rendering_config(config_name)
    except:
        config = get_bev_rendering_config(config_name)

    # if not (isinstance(config, EgoviewRenderingConfig) or isinstance(config, BevRenderingConfig)):
    #     raise RuntimeError("Invalid config.")

    if not config:
        raise RuntimeError("Invalid config.")

    return config
