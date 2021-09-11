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
Unit tests on methods exposed in the LocalVectorMap.
"""

from typing import List, Optional

import numpy as np
from argoverse.map_representation.map_api_v2 import (
    ArgoverseStaticMapV2,
    LaneSegment,
    LaneMarkType,
    LaneType,
    LocalLaneMarking,
    Point,
    Polyline,
)

import tbv.rendering.map_rendering_classes as map_rendering_classes
from tbv.rendering.map_rendering_classes import LocalVectorMap


def test_LocalVectorMap() -> None:
    """Ensure constructor call works without crashing, and that attributes can be retrieved correctly.

    "Fringe" is defined as being near the border of the BEV image canvas (i.e. would likely be cropped
    out during a random or center crop).

    20x20 scenario, i.e. we dilate 10 m in all directions (w/ infinity norm) from (2,2) which is AV location.
    """
    lvm = LocalVectorMap(avm=None, ego_center=np.array([2.0, 2.0]), dilation=10.0)

    llm1 = LocalLaneMarking(
        mark_type="DASHED_WHITE",
        src_lane_id=0,
        bound_side="left",
        polyline=np.array([[1, 5], [5, 1]]).astype(np.float32),
    )

    llm2 = LocalLaneMarking(
        mark_type="NONE",
        src_lane_id=0,
        bound_side="right",
        polyline=np.array([[2, 8], [8, 8]]).astype(np.float32),  # fringe bc 6 meters is greater than 10 - 5
    )

    llm3 = LocalLaneMarking(
        mark_type="SOLID_YELLOW",
        src_lane_id=1,
        bound_side="left",
        polyline=np.array([[-4, 1], [-4, 3]]).astype(np.float32),  # fringe bc 6 meters is greater than 10 - 5
    )
    lvm.nearby_lane_markings.extend([llm1, llm2, llm3])
    nf_ids = lvm.get_nonfringe_marking_idxs()
    assert nf_ids == [0]

    p_ids = lvm.get_painted_marking_idxs()
    assert p_ids == [0, 2]

    nf_p_ids = lvm.get_painted_nonfringe_marking_idxs()
    assert nf_p_ids == [0]


def test_sample_lane_id_biased_to_intersections() -> None:
    """Verify that we can correctly sample lane segments found inside intersections."""

    def make_dummy_lane_segment(id: int, is_intersection: bool) -> LaneSegment:
        """ """
        DUMMY_LANE_TYPE = LaneType.VEHICLE
        DUMMY_RIGHT_LANE_BOUNDARY = Polyline([Point(1, 2, 3), Point(4, 5, 6)])
        DUMMY_LEFT_LANE_BOUNDARY = Polyline([Point(1, 2, 3), Point(4, 5, 6)])
        DUMMY_RIGHT_MARK_TYPE = LaneMarkType.NONE
        DUMMY_LEFT_MARK_TYPE = LaneMarkType.SOLID_YELLOW
        DUMMY_SUCCESSORS = []

        return LaneSegment(
            id=id,
            is_intersection=is_intersection,
            lane_type=DUMMY_LANE_TYPE,
            right_lane_boundary=DUMMY_RIGHT_LANE_BOUNDARY,
            left_lane_boundary=DUMMY_LEFT_LANE_BOUNDARY,
            right_mark_type=DUMMY_RIGHT_MARK_TYPE,
            left_mark_type=DUMMY_LEFT_MARK_TYPE,
            successors=DUMMY_SUCCESSORS,
        )

    lvm = LocalVectorMap(avm=None, ego_center=np.zeros(2), dilation=20)
    lvm.nearby_lane_segment_dict[0] = make_dummy_lane_segment(id=0, is_intersection=False)
    lvm.nearby_lane_segment_dict[1] = make_dummy_lane_segment(id=1, is_intersection=False)
    lvm.nearby_lane_segment_dict[2] = make_dummy_lane_segment(id=2, is_intersection=True)

    np.random.seed(0)
    sampled_lane_id = lvm.sample_lane_id_biased_to_intersections()
    print(sampled_lane_id)
    assert sampled_lane_id == 2


def test_get_random_lane_id_sequence() -> None:
    """Ensure we can sample a valid, random sequence of chained lane segments.

    Note: predecessors are implicit, by manually inverting the DAG.

    Lane graph topology for test case scenario:

    5 \\         /--> 0 -> 7
       > 4 -> 3 ----> 1 -> 8
    6 /          \\-> 2 -> 9
    """

    def make_dummy_lane_segment(id: int, successors: List[int]) -> LaneSegment:
        """ """
        DUMMY_IS_INTERSECTION = False
        DUMMY_LANE_TYPE = LaneType.VEHICLE
        DUMMY_RIGHT_LANE_BOUNDARY = Polyline([Point(1, 2, 3), Point(4, 5, 6)])
        DUMMY_LEFT_LANE_BOUNDARY = Polyline([Point(1, 2, 3), Point(4, 5, 6)])
        DUMMY_RIGHT_MARK_TYPE = LaneMarkType.NONE
        DUMMY_LEFT_MARK_TYPE = LaneMarkType.SOLID_YELLOW

        return LaneSegment(
            id=id,
            is_intersection=DUMMY_IS_INTERSECTION,
            lane_type=DUMMY_LANE_TYPE,
            right_lane_boundary=DUMMY_RIGHT_LANE_BOUNDARY,
            left_lane_boundary=DUMMY_LEFT_LANE_BOUNDARY,
            right_mark_type=DUMMY_RIGHT_MARK_TYPE,
            left_mark_type=DUMMY_LEFT_MARK_TYPE,
            successors=successors,
        )

    lvm = LocalVectorMap(avm=None, ego_center=np.zeros(2), dilation=20)
    lvm.nearby_lane_segment_dict = {
        5: make_dummy_lane_segment(id=5, successors=[4]),
        6: make_dummy_lane_segment(id=6, successors=[4]),
        4: make_dummy_lane_segment(id=4, successors=[3]),
        3: make_dummy_lane_segment(id=3, successors=[0, 1, 2]),
        0: make_dummy_lane_segment(id=0, successors=[7]),
        1: make_dummy_lane_segment(id=1, successors=[8]),
        2: make_dummy_lane_segment(id=2, successors=[9]),
        7: make_dummy_lane_segment(id=7, successors=[]),
        8: make_dummy_lane_segment(id=8, successors=[]),
        9: make_dummy_lane_segment(id=9, successors=[]),
    }

    np.random.seed(0)
    traj_ids = lvm.get_random_lane_id_sequence(start_lane_id=4, path_length=3)
    assert traj_ids == [4, 3, 0]

    # choose random start id
    traj_ids = lvm.get_random_lane_id_sequence(path_length=3)
    assert traj_ids == [1, 8]  # hit leaf of graph, cannot iterate deeper

    # start at the end (cannot reach full length)
    traj_ids = lvm.get_random_lane_id_sequence(start_lane_id=0, path_length=3)
    assert traj_ids == [0, 7]

    # ensure we dont exceed the path length
    traj_ids = lvm.get_random_lane_id_sequence(start_lane_id=5, path_length=3)
    assert traj_ids == [5, 4, 3]


def test_get_random_lane_id_sequence_out_of_map() -> None:
    """Test when lane ids have already been pruned down before insertion."""

    def make_dummy_lane_segment(id: int, successors: List[int]) -> LaneSegment:
        """ """
        DUMMY_IS_INTERSECTION = False
        DUMMY_LANE_TYPE = LaneType.VEHICLE
        DUMMY_RIGHT_LANE_BOUNDARY = Polyline([Point(1, 2, 3), Point(4, 5, 6)])
        DUMMY_LEFT_LANE_BOUNDARY = Polyline([Point(1, 2, 3), Point(4, 5, 6)])
        DUMMY_RIGHT_MARK_TYPE = LaneMarkType.NONE
        DUMMY_LEFT_MARK_TYPE = LaneMarkType.SOLID_YELLOW

        return LaneSegment(
            id=id,
            is_intersection=DUMMY_IS_INTERSECTION,
            lane_type=DUMMY_LANE_TYPE,
            right_lane_boundary=DUMMY_RIGHT_LANE_BOUNDARY,
            left_lane_boundary=DUMMY_LEFT_LANE_BOUNDARY,
            right_mark_type=DUMMY_RIGHT_MARK_TYPE,
            left_mark_type=DUMMY_LEFT_MARK_TYPE,
            successors=successors,
        )

    lvm = LocalVectorMap(avm=None, ego_center=np.zeros(2), dilation=20)
    lvm.nearby_lane_segment_dict = {
        5: make_dummy_lane_segment(id=5, successors=[4]),
        4: make_dummy_lane_segment(id=4, successors=[999]),
    }
    # ensure we dont try to index into map at non-existent lane ID
    traj_ids = lvm.get_random_lane_id_sequence(start_lane_id=5, path_length=3)
    assert traj_ids == [5, 4]

    assert lvm.lane_is_in_map(5)
    assert lvm.lane_is_in_map(4)
    assert not lvm.lane_is_in_map(3)
    assert not lvm.lane_is_in_map(999)


def test_get_rightmost_neighbor() -> None:
    """Ensure we can correctly fetch the ID of the rightmost neighbor lane segment.

    Lane graph topology of test case scenario -- 4-lane road, with 2 lanes on both sides of the road:
    |7 |6 || 2 | 3 |
    |5 |4 || 0 | 1 |

    TODO: verify neighbors across road divide are labeled properly.
    """

    def make_dummy_lane_segment(
        id: int, successors: List[int], right_neighbor_id: Optional[int], left_neighbor_id: Optional[int]
    ) -> LaneSegment:
        """ """
        DUMMY_IS_INTERSECTION = False
        DUMMY_LANE_TYPE = LaneType.VEHICLE
        DUMMY_RIGHT_LANE_BOUNDARY = Polyline([Point(1, 2, 3), Point(4, 5, 6)])
        DUMMY_LEFT_LANE_BOUNDARY = Polyline([Point(1, 2, 3), Point(4, 5, 6)])
        DUMMY_RIGHT_MARK_TYPE = LaneMarkType.NONE
        DUMMY_LEFT_MARK_TYPE = LaneMarkType.SOLID_YELLOW

        return LaneSegment(
            id=id,
            is_intersection=DUMMY_IS_INTERSECTION,
            lane_type=DUMMY_LANE_TYPE,
            right_lane_boundary=DUMMY_RIGHT_LANE_BOUNDARY,
            left_lane_boundary=DUMMY_LEFT_LANE_BOUNDARY,
            right_mark_type=DUMMY_RIGHT_MARK_TYPE,
            left_mark_type=DUMMY_LEFT_MARK_TYPE,
            successors=successors,
            right_neighbor_id=right_neighbor_id,
            left_neighbor_id=left_neighbor_id,
        )

    lvm = LocalVectorMap(avm=None, ego_center=np.zeros(2), dilation=20)
    lvm.nearby_lane_segment_dict = {
        0: make_dummy_lane_segment(id=0, successors=[2], right_neighbor_id=1, left_neighbor_id=4),
        1: make_dummy_lane_segment(id=1, successors=[3], right_neighbor_id=None, left_neighbor_id=0),
        2: make_dummy_lane_segment(id=2, successors=[], right_neighbor_id=3, left_neighbor_id=6),
        3: make_dummy_lane_segment(id=3, successors=[], right_neighbor_id=None, left_neighbor_id=2),
        4: make_dummy_lane_segment(id=4, successors=[], right_neighbor_id=5, left_neighbor_id=0),
        5: make_dummy_lane_segment(id=5, successors=[], right_neighbor_id=None, left_neighbor_id=4),
        6: make_dummy_lane_segment(id=6, successors=[4], right_neighbor_id=7, left_neighbor_id=2),
        7: make_dummy_lane_segment(id=7, successors=[5], right_neighbor_id=None, left_neighbor_id=6),
    }

    query_lane_id = 2
    rightmost_lane_id = lvm.get_rightmost_neighbor(query_lane_id)
    assert rightmost_lane_id == 3

    query_lane_id = 3
    rightmost_lane_id = lvm.get_rightmost_neighbor(query_lane_id)
    assert rightmost_lane_id == 3

    query_lane_id = 7
    rightmost_lane_id = lvm.get_rightmost_neighbor(query_lane_id)
    assert rightmost_lane_id == 7

    query_lane_id = 6
    rightmost_lane_id = lvm.get_rightmost_neighbor(query_lane_id)
    assert rightmost_lane_id == 7


def test_get_nonfringe_lane_ids() -> None:
    """Ensure we can fetch the IDs of the lane segments not located near the scene boundary."""

    def make_dummy_lane_segment(id: int, right_lane_boundary: Polyline, left_lane_boundary: Polyline) -> LaneSegment:
        """ """
        DUMMY_IS_INTERSECTION = False
        DUMMY_LANE_TYPE = LaneType.VEHICLE
        DUMMY_RIGHT_LANE_BOUNDARY = Polyline([Point(1, 2, 3), Point(4, 5, 6)])
        DUMMY_LEFT_LANE_BOUNDARY = Polyline([Point(1, 2, 3), Point(4, 5, 6)])
        DUMMY_RIGHT_MARK_TYPE = LaneMarkType.NONE
        DUMMY_LEFT_MARK_TYPE = LaneMarkType.SOLID_YELLOW
        DUMMY_SUCCESSORS = []

        return LaneSegment(
            id=id,
            is_intersection=DUMMY_IS_INTERSECTION,
            lane_type=DUMMY_LANE_TYPE,
            right_lane_boundary=right_lane_boundary,
            left_lane_boundary=left_lane_boundary,
            right_mark_type=DUMMY_RIGHT_MARK_TYPE,
            left_mark_type=DUMMY_LEFT_MARK_TYPE,
            successors=DUMMY_SUCCESSORS,
        )

    lvm = LocalVectorMap(avm=None, ego_center=np.array([1, 0.0]), dilation=7.0)
    vls0 = make_dummy_lane_segment(
        id=0,
        right_lane_boundary=Polyline(waypoints=[Point(-4, 7, 0), Point(-4, 4, 0)]),
        left_lane_boundary=Polyline(waypoints=[Point(-3, 7, 0), Point(-3, 4, 0)]),
    )
    vls1 = make_dummy_lane_segment(
        id=1,
        right_lane_boundary=Polyline(waypoints=[Point(-4, 4, 0), Point(-4, 1, 0)]),
        left_lane_boundary=Polyline(waypoints=[Point(-3, 4, 0), Point(-3, 1, 0)]),
    )
    vls2 = make_dummy_lane_segment(
        id=2,
        right_lane_boundary=Polyline(waypoints=[Point(2, 1, 0), Point(2, 4, 0)]),
        left_lane_boundary=Polyline(waypoints=[Point(1, 1, 0), Point(1, 4, 0)]),
    )
    vls3 = make_dummy_lane_segment(
        id=3,
        right_lane_boundary=Polyline(waypoints=[Point(-2, -2, 0), Point(4, -2, 0)]),
        left_lane_boundary=Polyline(waypoints=[Point(-2, -1, 0), Point(4, -1, 0)]),
    )
    lvm.nearby_lane_segment_dict = {0: vls0, 1: vls1, 2: vls2, 3: vls3}
    nonfringe_lane_ids = lvm.get_nonfringe_lane_ids()
    assert nonfringe_lane_ids == [2, 3]


def test_normalize_prob_distribution() -> None:
    """Ensure a 1d vector of scores can be normalized to a valid probability distribution."""
    probs = np.array([2.0, 2.0, 1.0])
    norm_probs = map_rendering_classes.normalize_prob_distribution(probs)
    gt_norm_probs = np.array([0.4, 0.4, 0.2])
    assert np.allclose(norm_probs, gt_norm_probs)
