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
Interface to fetch chained lane segments, lane markings not next to "scene" boundary (e.g. 20 meters from ego-vehicle)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import av2.geometry.infinity_norm_utils as infinity_norm_utils
import av2.geometry.interpolate as interp_utils
import numpy as np
from av2.map.lane_segment import LocalLaneMarking
from av2.map.map_api import ArgoverseStaticMap

from tbv.common.tbv_lane_segment import TbvLaneSegment


PED_XING_INTERSECTION_PROB = 0.9
PED_XING_NONINTERSECTION_PROB = 0.2
WPT_INFTY_NORM_INTERP_NUM = 50

# 5 meters away from image boundary is considered the "fringe"
# "Fringe" is defined as being near the border of the BEV image canvas (i.e. would likely be cropped
# out during a random or center crop).
EGOVEHICLE_TO_FRINGE_DIST_M = 5


class LocalPedCrossing(NamedTuple):
    """Represents a pedestrian crossing (i.e. crosswalk) as two edges along its principal axis.

    Both lines should be pointing in nominally the same direction and a pedestrian is expected to
    move either roughly parallel to both lines or anti-parallel to both lines.

    Args:
        edge1: array of shape (2,3) representing one edge of the crosswalk, with 2 waypoints (3d polyline)
        edge2: array of shape (2,3) representing the other edge of the crosswalk, with 2 waypoints (3d polyline).
        is_nearby:
    """

    edge1: np.ndarray
    edge2: np.ndarray
    is_nearby: bool = True  # assume new synthetic xwalk will be nearby, who cares if not deleting

    def get_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        return (self.edge1[:, :2], self.edge2[:, :2])


class LocalVectorMap:
    def __init__(self, avm: ArgoverseStaticMap, ego_center: np.ndarray, range_m: float) -> None:
        """Class to assist with vector map perturbation, using knowledge of the lane graph topology.

        Note: In the ego-view, we use the `changed_points` attribute to determine whether or not a
        synthetic change is visible in an image.

        Args:
            avm: local map for TbV log/scenario.
            ego_center: array of shape (x,y) representing the 2d coordinates of the ego-vehicle/AV in city frame.
            range_m: maximum distance in l-infinity norm from origin (egovehicle) for entities that will be rendered.
        """
        self.avm: ArgoverseStaticMap = avm
        self.ped_crossing_edges: List[LocalPedCrossing] = []
        self.stoplines: List[np.ndarray] = []
        self.nearby_lane_segment_dict: Dict[int, TbvLaneSegment] = {}
        self.nearby_lane_markings: List[LocalLaneMarking] = []
        self.ego_center: np.ndarray = ego_center
        self.range_m: float = range_m
        self.changed_points: Optional[np.ndarray] = None

    def get_nonfringe_lane_ids(self) -> List[int]:
        """
        TODO: use shapely poiint-polygon distance instead, if faster!
        """
        return [
            lane_id
            for lane_id, vls in self.nearby_lane_segment_dict.items()
            if vls.is_within_l_infinity_norm_radius(
                query_center=self.ego_center, search_radius_m=(self.range_m - EGOVEHICLE_TO_FRINGE_DIST_M)
            )
        ]

    def get_nonfringe_marking_idxs(self) -> List[int]:
        """Fetch indices of all lane markings in list that are not located near scene boundary."""
        is_not_fringe = np.zeros(len(self.nearby_lane_markings), dtype=bool)
        for i, llm in enumerate(self.nearby_lane_markings):
            try:
                interp_polyline = interp_utils.interp_arc(WPT_INFTY_NORM_INTERP_NUM, points=llm.polyline[:, :2])
            except Exception as e:
                print("Interpolation failed")
                logging.exception("Interpolation failed")
                interp_polyline = llm.polyline
            is_not_fringe[i] = infinity_norm_utils.has_pts_in_infinity_norm_radius(
                interp_polyline, self.ego_center, self.range_m - EGOVEHICLE_TO_FRINGE_DIST_M
            )

        return list(np.where(is_not_fringe)[0])

    def get_painted_marking_idxs(self) -> List[int]:
        """Fetch indices of lane markings in list that correspond to painted (i.e. white or yellow) lane markings."""
        # check which boundaries have painted lane markings
        is_painted_bnd = [llm.mark_type != "NONE" for llm in self.nearby_lane_markings]
        is_painted_bnd = np.array(is_painted_bnd)
        return list(np.where(is_painted_bnd)[0])

    def get_painted_nonfringe_marking_idxs(self) -> List[int]:
        """Fetch indices of lane markings in list that correspond to painted (i.e. white or yellow) lane markings
        that are not located near scene boundary.
        """
        return list(set(self.get_nonfringe_marking_idxs()).intersection(self.get_painted_marking_idxs()))

    def sample_lane_id_biased_to_intersections(self) -> int:
        """Sample id of a lane segment from a distribution biased towards lane segments found in intersections.

        Weight according to distance to nearest intersection lane segment to increase realism (9:2 ratio)
        """
        lane_ids_to_sample = []
        sample_probs = []
        for lane_id, vls in self.nearby_lane_segment_dict.items():
            lane_ids_to_sample.append(lane_id)
            p = PED_XING_INTERSECTION_PROB if vls.is_intersection else PED_XING_NONINTERSECTION_PROB
            sample_probs.append(p)

        sample_probs = np.array(sample_probs)
        sample_probs = normalize_prob_distribution(sample_probs)
        sampled_lane_id = np.random.choice(lane_ids_to_sample, p=sample_probs)

        # Alternative strategy: check if better to be inside intersection, or at most 1 away from intersection lane
        return sampled_lane_id

    def lane_is_in_map(self, lane_id: int) -> bool:
        """Check whether a lane ID is located inside the local vector map."""
        return lane_id in self.nearby_lane_segment_dict.keys()

    def get_random_lane_id_sequence(self, start_lane_id: Optional[int] = None, path_length: int = 2) -> List[int]:
        """Depth first-search up to desired graph, with random exploration.

        Args:
            start_lane_id: unique ID of lane segment to start sampling the path/trajectory of lane segments at.
                If not provided, any random lane ID will be chosen.
            path_length: maximum possible length of path/trajectory (number of consecutive lane segments).
                Note: path_length may not be achievable, based on lane graph topology.

        Returns:
            traj_lane_ids: path/trajectory of chained, consecutive lane segments.
        """
        if start_lane_id is None:
            start_lane_id = np.random.choice(list(self.nearby_lane_segment_dict.keys()))

        traj_lane_ids = [start_lane_id]
        # begin iterating through the graph
        curr_lane_id = start_lane_id

        # try to achieve the path length, if possible. Note: we cannot proceed deeper if successor not in map.
        while (
            len(traj_lane_ids) < path_length
            and len(self.nearby_lane_segment_dict[curr_lane_id].successors) > 0
            and all([self.lane_is_in_map(succ) for succ in self.nearby_lane_segment_dict[curr_lane_id].successors])
        ):
            curr_lane_id = np.random.choice(self.nearby_lane_segment_dict[curr_lane_id].successors)
            traj_lane_ids += [curr_lane_id]

        return traj_lane_ids

    def get_rightmost_neighbor(self, curr_lane_id: int) -> int:
        """Fetch lane ID of rightmost neighbor of a specified lane segment ID.

        Implemented as: keep looping through right neighbors until there are no more.
        """
        while self.nearby_lane_segment_dict[curr_lane_id].right_neighbor_id is not None:
            curr_lane_id = self.nearby_lane_segment_dict[curr_lane_id].right_neighbor_id
        return curr_lane_id


def normalize_prob_distribution(probs: np.ndarray) -> np.ndarray:
    """Normalize a 1d vector of logits/scores to a probability distribution.

    Args:
        probs: array of shape (N,) representing an unnormalized probability distribution.

    Returns:
        probs: array of shape (N,) representing an normalized probability distribution.
    """
    probs /= probs.sum()
    return probs
