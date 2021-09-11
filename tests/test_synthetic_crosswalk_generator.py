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
Unit tests to ensure that utilities for generating synthetic crosswalks work properly.
"""

import numpy as np
from shapely.geometry import LineString

import tbv.synthetic_generation.synthetic_crosswalk_generator as synthetic_crosswalk_generator
from tbv.synthetic_generation.synthetic_crosswalk_generator import Line2D


def test_extend_linestring() -> LineString:
    """Ensure line segments (as LineStrings) can be extended outwards."""
    # vertical case
    polyline = np.array([[4, 4], [4, 12]])
    ls = LineString(polyline)
    e_ls = synthetic_crosswalk_generator.extend_linestring(ls)
    # a line segment of length 8 from (4,4) <-> (4,12) would become (4,4-800) <-> (4,12+800).
    assert np.allclose(np.array(e_ls.coords), np.array([[4.0, -796.0], [4.0, 812.0]]))

    # horizontal case
    polyline = np.array([[4, 4], [12, 4]])
    ls = LineString(polyline)
    e_ls = synthetic_crosswalk_generator.extend_linestring(ls)
    # a line segment of length 8 from (4,4) <-> (12,4) would become (4-800,4) <-> (12+800,4).
    assert np.allclose(np.array(e_ls.coords), np.array([[-796.0, 4.0], [812.0, 4.0]]))

    # diagonal case
    polyline = np.array([[4, 4], [10, 10]])
    ls = LineString(polyline)
    e_ls = synthetic_crosswalk_generator.extend_linestring(ls)
    # a line segment from (4,4) <-> (10,10) would become (4-600,4-600) <-> (10+600,10+600).
    assert np.allclose(np.array(e_ls.coords), np.array([[-596.0, -596.0], [610.0, 610.0]]))


def test_build_random_crosswalk_from_edge_horizontal() -> None:
    """Ensure that a crosswalk can be generated from its right edge (when horizontal) and a desired crosswalk width.

    Note: sample 2nd edge is to the left of the first provided edge. Width should be 2 meters.

    Scenario: Given A->B (right polyline), we would generate C->D as follows (left polyline)
        C ---> D

        A ---> B
    """
    # fmt: off
    right_polyline = np.array(
    	[
    		[4, 4],
    		[12, 4]
    	]
    )
    # fmt: on
    ped_xing_edges = synthetic_crosswalk_generator.build_random_crosswalk_from_edge(right_polyline, cw_width=2)
    # fmt: off
    gt_ped_xing_edges = (
    	np.array([[4, 4], [12, 4]]), # right edge
    	np.array([[4.0, 6.0], [12.0, 6.0]]) # left edge
    )
    # fmt: on
    assert isinstance(ped_xing_edges, tuple)
    assert len(ped_xing_edges) == 2
    assert np.array_equal(ped_xing_edges[0], gt_ped_xing_edges[0])
    assert np.array_equal(ped_xing_edges[1], gt_ped_xing_edges[1])


def test_build_random_crosswalk_from_edge_vertical() -> None:
    """Ensure that a crosswalk can be generated from its right edge (when vertical) and a desired crosswalk width.

    Note: sample 2nd edge is to the left of the first provided edge. Width should be 2 meters.

    Scenario: Given A->B (right polyline), we would generate C->D as follows (left polyline)
        D    B
        /\\ /\
	    ||  ||
        ||  ||
        C   A
    """
    # fmt: off
    right_polyline = np.array(
        [
            [4, 4],
            [4, 12]
        ]
    )
    # fmt: on
    ped_xing_edges = synthetic_crosswalk_generator.build_random_crosswalk_from_edge(right_polyline, cw_width=2)
    # fmt: off
    gt_ped_xing_edges = (
        np.array([[4, 4], [4, 12]]),
        np.array([[2.0, 4.0], [2.0, 12.0]])
    )
    # fmt: on
    assert isinstance(ped_xing_edges, tuple)
    assert len(ped_xing_edges) == 2
    assert np.array_equal(ped_xing_edges[0], gt_ped_xing_edges[0])
    assert np.array_equal(ped_xing_edges[1], gt_ped_xing_edges[1])


def test_build_random_crosswalk_from_edge_diagonal() -> None:
    """Ensure that a crosswalk can be generated from its right edge (when diagonal) and a desired crosswalk width."""
    # fmt: off
    right_polyline = np.array(
        [
            [4, 4],
            [10, 10]
        ]
    )
    # fmt: on
    ped_xing_edges = synthetic_crosswalk_generator.build_random_crosswalk_from_edge(right_polyline, cw_width=2)
    # fmt: off

    # since sqrt(2) = 1.41, we move left 1.41 and up by 1.41
    gt_ped_xing_edges = (
        np.array([[4, 4], [10, 10]]),
        np.array([[2.59, 5.41], [8.59, 11.41]])
    )
    # fmt: on
    assert isinstance(ped_xing_edges, tuple)
    assert len(ped_xing_edges) == 2
    assert np.allclose(ped_xing_edges[0], gt_ped_xing_edges[0])
    assert np.allclose(ped_xing_edges[1], gt_ped_xing_edges[1], atol=0.01)


def test_find_xwalk_iou_2d() -> None:
    """Ensure that crosswalk IoU can be computed properly.

    Two crosswalks, both with area 8, that overlap at 4 grid cells, means union 12, and intersection 4.

    right and left polylines for crosswalk 1 -> R1 and L1

    O   O
    x-------x L1
    |   |
    x-------x R1
    O   O
    L2  R2
    """

    # fmt: off
    xwalk_1 = (
        np.array([[2, 1], [6, 1]]),
        np.array([[2, 3], [6, 3]]))

    xwalk_2 = (
        np.array([[4, 0], [4, 4]]),
        np.array([[2, 0], [2, 4]])
    )
    # fmt: on
    iou = synthetic_crosswalk_generator.find_xwalk_iou_2d(xwalk_1, xwalk_2)
    assert np.isclose(4 / 12, iou)


def test_find_1d_closest_idx_to_origin() -> None:
    """Ensure that the index of the value closest to zero on either side (pos. or neg.) can be found correctly."""
    x = np.array([0.04163795, 0.02760515, 0.01410576, -0.00184245])
    pos_idx = synthetic_crosswalk_generator.find_1d_closest_idx_to_origin(x, "positive")

    # 0.014 is the smallest positive value
    assert pos_idx == 2
    neg_idx = synthetic_crosswalk_generator.find_1d_closest_idx_to_origin(x, "negative")
    # -0.0018 is the largest negative value
    assert neg_idx == 3


def test_get_extended_normal_linestring() -> None:
    """Ensure that we can generate a long line segment along the direction of a line's normal.

    Line's normal is the vector (0,1).
    """
    a, b, c = 0, 1, 3
    line = Line2D(a, b, c)
    assert isinstance(line, Line2D)
    assert line.a == 0
    assert line.b == 1
    assert line.c == 3

    center_pt = np.array([-1, -3])
    extended_ls = line.get_extended_normal_linestring(center_waypt=center_pt)

    # normal has length 1, so we get a length 2 line segment (1 meter in each direction).
    # Then we extend the line segment by 100x its length in each direction (200 m. up and 200 m. down.).

    # -2 + 200=198 and -4 - 200 = -204
    pos_dir_pt = (-1.0, 198.0)
    neg_dir_pt = (-1.0, -204.0)

    assert list(extended_ls.coords) == [pos_dir_pt, neg_dir_pt]


# def test_get_closest_pt() -> None:
#     """ """
#     a, b, c = 0, 1, 3
#     line = Line2D(a, b, c)

#     pts = np.array([[1, -2], [2, -1], [2, -4], [-1, -5]])
#     closest_pos_pt = line.get_closest_pt(pts, direction="positive")
#     gt_closest_pos_pt = np.array([1, -2])
#     assert np.allclose(closest_pos_pt, gt_closest_pos_pt)

#     closest_neg_pt = line.get_closest_pt(pts, direction="negative")
#     gt_closest_neg_pt = np.array([2, -4])
#     assert np.allclose(closest_neg_pt, gt_closest_neg_pt)


def test_get_point_signed_dists_to_line_diagonal_line() -> None:
    """Ensure that point-line distances are computed accurately, for line y=x.

    Two points indicated by X:

     X | /line 1
       |/
    --------
     / |
    /  |
       X
    """
    pts = np.array([[-1, 1], [0, -2]])

    # normal is in (1,-1) direction \\
    a = 1
    b = -1
    c = 0
    line1 = Line2D(a, b, c)
    dists = line1.get_point_signed_dists_to_line(pts)
    # each point should be sqrt(2) units away.
    gt_dists = np.array([-np.sqrt(2), np.sqrt(2)])
    assert np.allclose(dists, gt_dists)


def test_get_point_signed_dists_to_line_vertical_line() -> None:
    """Ensure that point-line distances are computed accurately, for line y=-3.

    Two points indicated by X:

     X |
       |
    --------
       |
       |
       X
    - - - - - - line 2
    """
    pts = np.array([[-1, 1], [0, -2]])
    # normal is in (0,1) || direction
    a = 0
    b = 1
    c = 3

    line2 = Line2D(a, b, c)
    dists = line2.get_point_signed_dists_to_line(pts)
    gt_dists = np.array([4, 1])
    assert np.allclose(dists, gt_dists)


def test_get_polyline_normal_2d_straight() -> None:
    """Ensure that we can compute the normal to a polyline at waypoint with index=2. All points along line y=-3.

                   //\
                    |
    ---> rotated to |

    """
    # fmt: off
    polyline_pts = np.array(
        [
            [-2, -3],
            [-1, -3],
            [0, -3],
            [2, -3]
        ]
    )
    # fmt: on
    normal_line_2d = synthetic_crosswalk_generator.get_polyline_normal_2d(polyline_pts, waypt_idx=2)
    assert np.isclose(normal_line_2d.a, 0)
    assert np.isclose(normal_line_2d.b, 1)
    assert np.isclose(normal_line_2d.c, 3)


def test_get_polyline_normal_2d_piecewise() -> None:
    """Ensure that we can compute the normal to a polyline at waypoint with index=2.

    Polyline has shape:
    |
    \\
     \\
       ---
    """
    # fmt: off
    polyline_pts = np.array(
        [
            [-1,3],
            [-1,1],
            [0,0], # find approximation here.
            [1,-1],
            [3,-1]
        ]
    )
    # fmt: on
    normal_line_2d = synthetic_crosswalk_generator.get_polyline_normal_2d(polyline_pts, waypt_idx=2)
    assert np.isclose(normal_line_2d.a, 1 / np.sqrt(2))
    assert np.isclose(normal_line_2d.b, 1 / np.sqrt(2))
    assert np.isclose(normal_line_2d.c, 0)
