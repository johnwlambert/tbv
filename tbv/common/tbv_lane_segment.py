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

from av2.map.lane_segment import LaneSegment


class TbvLaneSegment(LaneSegment):
    """Container that holds state of whether a lane segment should be rendered.

    Note: the `predecessors` attribute of LaneSegment is not used anywhere within TbV (only successors are used).

    Args:
        render_l_bound: boolean flag for visualization, indicating whether to render the left lane boundary.
        render_r_bound: boolean flag for visualization, indicating whether to render the right lane boundary.
    """

    render_l_bound: Optional[bool] = True
    render_r_bound: Optional[bool] = True

    @classmethod
    def from_lane_segment(cls, ls: LaneSegment) -> "TbvLaneSegment":
        """Construct instance from existing LaneSegment object."""
        return cls(
            id=ls.id,
            is_intersection=ls.is_intersection,
            lane_type=ls.lane_type,
            right_lane_boundary=ls.right_lane_boundary,
            left_lane_boundary=ls.left_lane_boundary,
            right_mark_type=ls.right_mark_type,
            left_mark_type=ls.left_mark_type,
            predecessors=ls.predecessors,
            successors=ls.successors,
            right_neighbor_id=ls.right_neighbor_id,
            left_neighbor_id=ls.left_neighbor_id,
        )
