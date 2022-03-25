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

import numpy as np
from av2.map.lane_segment import LaneMarkType, LaneSegment, LaneType

from tbv.common.tbv_lane_segment import TbvLaneSegment


def test_tbv_lane_segment() -> None:
    """ """
    json_data = {
        "id": 93269421,
        "is_intersection": False,
        "lane_type": "VEHICLE",
        "left_lane_boundary": [
            {"x": 873.97, "y": -101.75, "z": -19.7},
            {"x": 880.31, "y": -101.44, "z": -19.7},
            {"x": 890.29, "y": -100.56, "z": -19.66},
        ],
        "left_lane_mark_type": "SOLID_YELLOW",
        "left_neighbor_id": None,
        "right_lane_boundary": [{"x": 874.01, "y": -105.15, "z": -19.58}, {"x": 890.58, "y": -104.26, "z": -19.58}],
        "right_lane_mark_type": "SOLID_WHITE",
        "right_neighbor_id": 93269520,
        "successors": [93269500],
        "predecessors": [],
    }
    lane_segment = LaneSegment.from_dict(json_data)
    tbv_ls = TbvLaneSegment.from_lane_segment(lane_segment)

    assert isinstance(tbv_ls, TbvLaneSegment)

    expected_right_lane_boundary = np.array([[874.01, -105.15, -19.58], [890.58, -104.26, -19.58]])
    assert np.array_equal(tbv_ls.right_lane_boundary.xyz, expected_right_lane_boundary)
    expected_left_lane_boundary = np.array(
        [[873.97, -101.75, -19.7], [880.31, -101.44, -19.7], [890.29, -100.56, -19.66]]
    )
    assert np.array_equal(tbv_ls.left_lane_boundary.xyz, expected_left_lane_boundary)

    assert tbv_ls.lane_type == LaneType.VEHICLE

    assert tbv_ls.id == 93269421
    assert not tbv_ls.is_intersection

    assert tbv_ls.successors == [93269500]
    assert tbv_ls.predecessors == []

    assert tbv_ls.left_mark_type == LaneMarkType.SOLID_YELLOW
    assert tbv_ls.right_mark_type == LaneMarkType.SOLID_WHITE

    assert tbv_ls.left_neighbor_id is None
    assert tbv_ls.right_neighbor_id == 93269520

    assert tbv_ls.render_l_bound
    assert tbv_ls.render_r_bound


if __name__ == "__main__":
    test_tbv_lane_segment()
