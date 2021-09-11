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

from typing import Tuple

import numpy as np

from tbv.utils.proj_utils import LogEgoviewRenderingMetadata
from tbv.utils.triangle_grid_utils import render_triangles_in_egoview
from tbv.utils.triangulation_2d import form_polygon_triangulation_2d


def render_polygon_egoview(
    ego_metadata: LogEgoviewRenderingMetadata,
    img_bgr: np.ndarray,
    polygon_pts_cityfr: np.ndarray,
    downsample_factor: float,
    allow_interior_only: bool,
    filter_to_driveable_area: bool,
    color_rgb: Tuple[int,int,int]
) -> np.ndarray:
    """Triangulate a polygon in 2d, then potentially filter the triangles based on a 2d raster grid, then
    append 3d coordinates. Then project triangles into a perspective camera and rasterize.

    Args:
        ego_metadata:
        img_bgr:
        polygon_pts_cityfr:
        downsample_factor:
        allow_interior_only:
        filter_to_driveable_area:
        color_rgb:
    """
    # region not well expressed near image borders if downsample_factor is not very small
    triangles = form_polygon_triangulation_2d(
        polygon_pts_cityfr,
        downsample_factor,
        allow_interior_only
    )

    if filter_to_driveable_area:
        verts_cityfr = triangles.reshape(-1,2)
        verts_cityfr = ego_metadata.avm.append_height_to_2d_city_pt_cloud(pt_cloud_xy=verts_cityfr)
        vert_inside_da = ego_metadata.avm.get_raster_layer_points_boolean(verts_cityfr, layer_name="driveable_area")
        tri_inside_da = vert_inside_da.reshape(-1,3).sum(axis=1) == 3
        da_triangles = triangles[tri_inside_da]
        triangles = da_triangles

    tri_verts_cityfr = triangles.reshape(-1,2)
    tri_verts_cityfr = ego_metadata.avm.append_height_to_2d_city_pt_cloud(pt_cloud_xy=tri_verts_cityfr)
    tri_verts_egofr = ego_metadata.city_SE3_egovehicle.inverse().transform_point_cloud(tri_verts_cityfr)

    img_bgr = render_triangles_in_egoview(
        ego_metadata.depth_map,
        img_bgr,
        tri_verts_egofr,
        ego_metadata.log_calib_data,
        ego_metadata.camera_name,
        color_rgb
    )
    return img_bgr

