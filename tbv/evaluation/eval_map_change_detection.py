# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Map change detection evaluation script."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import av2.geometry.polyline_utils as polyline_utils
import av2.utils.io as io_utils
import click
import numpy as np
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from rich.progress import track
from shapely.geometry import LinearRing, LineString, Point, Polygon


LABELED_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "labeled_data"

# interpolate a vertex every 10 cm along a changed map entity.
INTERP_INTERVAL_M = 0.10


DUAL_CATEGORY_DICT = {
    "delete_crosswalk": "insert_crosswalk",
    "insert_crosswalk": "delete_crosswalk",
    "lane_geometry_change": "lane_geometry_change"  # TODO: add fine-grained
    # 'change_lane_marking_color':'change_lane_marking_color'
    # 'delete_lane_marking': 4,
    # 'add_bike_lane': 5,
    # 'change_lane_boundary_dash_solid': 6
}


def point_to_line_dist(pt: np.ndarray, polyline: np.ndarray):
    """Compute the closest distance from a point to any point along a line.

    Args:
        polyline: (N,2) representing 2d polyline vertices.
        pt: (1,2) representing a 2d point.

    Returns:
        Shortest distance from point to polyline.
    """
    if polyline.ndim != 2:
        raise ValueError("Polyline must have shape (N,2)")

    if polyline.shape[1] != 2:
        raise ValueError("Polyline must have shape (N,2)")

    if pt.shape != (2,):
        raise ValueError("Point must have shape (2,)")

    line = LineString(polyline)
    p = Point(pt)
    dist = line.distance(p)
    return dist


def point_to_polygon_dist(pt: np.ndarray, polygon: np.ndarray, show_plot: bool = False) -> float:
    """Returns polygon to point distance

    Args:
        polygon: as a shapely Polygon
        pt: as a shapely Point
        show_plot: boolean indicating whether to visualize the objects using Matplotlib.

    Returns:
        dist: float representing distance ...
    """
    if polygon.ndim != 2:
        raise ValueError("Polygon must have shape (N,2)")

    if polygon.shape[1] != 2:
        raise ValueError("Polygon must have shape (N,2)")

    if pt.shape != (2,):
        raise ValueError("Point has shape (2,)")

    pt = Point(pt)
    polygon = Polygon(polygon)

    pol_ext = LinearRing(polygon.exterior.coords)
    dist = pol_ext.distance(pt)

    # # distance along the ring to a point nearest the other object.
    # d_geodesic = pol_ext.project(pt)
    # # return a point at the specified distance along the ring
    # nearest_p = pol_ext.interpolate(d_geodesic)
    # diff = np.array(nearest_p.coords) - np.array(pt)
    # dist = np.linalg.norm(diff)
    show_plot = False  # True
    if show_plot:
        import matplotlib.pyplot as plt

        pt_coords = np.array(pt.coords).squeeze()
        nearest_p_coords = np.array(nearest_p.coords).squeeze()

        plt.scatter(pt_coords[0], pt_coords[1], 10, color="k")
        plt.scatter(nearest_p_coords[0], nearest_p_coords[1], 10, color="r")
        print("Nearest p: ", nearest_p_coords)
        plt.plot(*polygon.exterior.xy, color="b")
        plt.axis("equal")
        plt.title(f"Dist = {dist:.1f}, nearest point: {np.round(nearest_p_coords, 1)}")
        plt.show()

    return dist


@dataclass(frozen=True)
class SpatialMapChangeEvent:
    """Object that stores a specific, spatially-localized area of a map that has undergone a real-world change.

    Args:
        log_id: unique ID of TbV vehicle log.
        change_type:
        city_coords: (N,3) array representing polyline (for lane line change)
            or polygon vertices (for crosswalk area change).
    """

    log_id: str
    change_type: str
    city_coords: np.ndarray

    def __post_init__(self) -> None:
        """Verify the shape."""
        if self.city_coords.ndim != 2:
            raise ValueError("Map change entity must be a 2d array, i.e. of shape (N,3).")

        if self.city_coords.shape[1] != 3:
            raise ValueError("Map change entity must be of shape (N,3).")

    @classmethod
    def from_event_dict(
        cls, log_id: str, event_dict: Dict[str, Any], avm: ArgoverseStaticMap
    ) -> "SpatialMapChangeEvent":
        """
        Args:
            cls:
            event_dict: dictionary data, loaded from JSON, containing information about map change event.

        Returns:
            New SpatialMapChange object.
        """
        if event_dict["supercategory"] not in ["crosswalk_change", "lane_geometry_change"]:
            raise ValueError("Unknown change type.")

        if event_dict["supercategory"] == "crosswalk_change":
            sensor_change_type = event_dict["change_type"]
            map_change_type = DUAL_CATEGORY_DICT[sensor_change_type]
            city_coords_2d = np.array(event_dict["polygon"])

        elif event_dict["supercategory"] == "lane_geometry_change":
            # print(log_id)
            sensor_change_type = event_dict["change_type"]
            map_change_type = DUAL_CATEGORY_DICT[sensor_change_type]
            map_change_type = "change_lane_marking_color"

            visible_line_segment = []
            for waypt_dict in event_dict["change_endpoints"]:
                waypt = np.array(waypt_dict["City Coords"])
                visible_line_segment += [waypt]
            visible_line_segment = np.array(visible_line_segment)
            assert visible_line_segment.shape[0] >= 2  # need at least 2 waypoints
            city_coords_2d = visible_line_segment

        # lift 2d coordinates to 3d.
        city_coords_3d = avm.append_height_to_2d_city_pt_cloud(points_xy=city_coords_2d)
        return cls(
            log_id=log_id,
            change_type=map_change_type,
            city_coords=city_coords_3d,
        )

    def check_if_in_range(self, log_id: str, city_SE3_egovehicle: SE3, range_thresh_m: float) -> bool:
        """Determine whether a changed map entity falls within some spatial distance from the egovehicle's pose.

        We use point-to-polygon distance for crosswalk, and point-to-line distance for line objects.

        Args:
            log_id: unique ID of TbV vehicle log.
            city_SE3_egovehicle: egovehicle's pose (in city coordinate frame) at timestamp of interest.
            range_thresh_m: maximum range (in meters) to use for evaluation. Map entities found
                beyond this distance will not be considered for evaluation.

        Returns:
            Whether ... (by L2 norm, not L-infinity norm).
        """
        if not isinstance(city_SE3_egovehicle, SE3):
            raise ValueError("Query egovehicle pose must be SE(3) object.")

        if "crosswalk" in self.change_type:
            # map entity represents a polygon.
            dist = point_to_polygon_dist(pt=city_SE3_egovehicle.translation[:2], polygon=self.city_coords[:, :2])
        else:
            # map entity represents a polyline.
            dist = point_to_line_dist(pt=city_SE3_egovehicle.translation[:2], polyline=self.city_coords[:, :2])

        return dist < range_thresh_m

    def check_if_in_range_egoview(
        self, log_id: str, city_SE3_egovehicle: SE3, range_thresh_m: float, pinhole_cam: PinholeCamera
    ) -> bool:
        """Determine whether a changed map entity is visibile within a camera's frustum, i.e. should have been detected

        Args:
            log_id: unique ID of TbV vehicle log.
            city_SE3_egovehicle: egovehicle's pose (in city coordinate frame) at timestamp of interest.
            range_thresh_m: maximum range (in meters) to use for evaluation. Map entities found
                beyond this distance will not be considered for evaluation.
            pinhole_cam: pinhole camera object, representing camera of interest.

        Returns:
            Boolean indicating a changed map entity is both (1) within spatial range, and (2) visible in camera frustum
        """
        if not isinstance(city_SE3_egovehicle, SE3):
            raise ValueError("Query egovehicle pose must be SE(3) object.")

        is_nearby = self.check_if_in_range(
            log_id=log_id, city_SE3_egovehicle=city_SE3_egovehicle, range_thresh_m=range_thresh_m
        )
        if not is_nearby:
            # if changed map entity is not nearby, immediately reject it.
            return False

        city_coords = self.city_coords
        if "crosswalk" in self.change_type:
            # complete last edge of the square (for polygon)
            city_coords = np.vstack([city_coords, city_coords[0]])

        # We interpolate points along map entity, e.g. if crosswalk endpoints are outside of the visible
        # field of view on the left of the image, and on the right of the image.
        # We sample one point for every 10 cm. we can't use a fixed number of points, as some polylines
        # are up to 500+ meters in length, meaning 50 samples would be insufficient.
        city_coords_interp, _ = polyline_utils.interp_polyline_by_fixed_waypt_interval(
            polyline=city_coords, waypt_interval=INTERP_INTERVAL_M
        )
        entity_egofr = city_SE3_egovehicle.inverse().transform_point_cloud(city_coords_interp)
        _, _, valid_pts_bool = pinhole_cam.project_ego_to_img(points_ego=entity_egofr, remove_nan=False)

        is_visible = valid_pts_bool.sum() > 0
        if not is_visible:
            logging.info(
                "None of the changed points "
                + str(np.round(entity_egofr.mean(axis=0)))
                + f"projected into frustum {pinhole_cam.cam_name}"
            )

        return is_visible


def get_test_set_event_info_bev(data_root: Path, split: str) -> Dict[str, List[SpatialMapChangeEvent]]:
    """Load GT labels for test set from disk, and convert them to SpatialMapChangeEvent objects per log.

    Args:
        data_root: path to local directory, where TbV logs are stored on disk.

    Returns:
        logid_to_mc_events_dict: dictionary from log_id to associated annotated GT map change events.
    """
    if split not in ["val", "test"]:
        raise ValueError("Cannot evaluate on a split other than val or test.")

    localization_data_fpath = LABELED_DATA_ROOT / f"tbv_{split}_split_annotations.json"

    # TODO: name this file by the split name.
    #localization_data_fpath = LABELED_DATA_ROOT / "mcd_test_set_localization_in_space.json"
    localization_data = io_utils.read_json_file(localization_data_fpath)

    logid_to_mc_events_dict = defaultdict(list)

    for log_data in track(localization_data, description="Loading map change annotations..."):
        log_id = log_data["log_id"]
        log_map_dirpath = data_root / log_id / "map"
        avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

        # init an empty list, in case it was a `positive` before or after log
        logid_to_mc_events_dict[log_id] = []
        for event_data in log_data["events"]:
            mc_event = SpatialMapChangeEvent.from_event_dict(log_id=log_id, event_dict=event_data, avm=avm)
            logid_to_mc_events_dict[log_id] += [mc_event]

    print(f"Loaded SpatialMapChangeEvents for {len(logid_to_mc_events_dict)} logs in {split} split.")
    return logid_to_mc_events_dict


@click.command(help="Evaluate predictions on the val or test split of the TbV Dataset.")
@click.option(
    "-d",
    "--data-root",
    required=True,
    help="Path to local directory where the TbV Dataset logs are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "--split",
    type=str,
    required=True
)
def run_evaluate_map_change_detection(data_root: str, split: str) -> None:
    """Click entry point for evaluation of map change detection results."""
    get_test_set_event_info_bev(data_root=Path(data_root), split=split)


if __name__ == "__main__":
    run_evaluate_map_change_detection()
