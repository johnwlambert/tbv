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
Unit tests to make sure crosswalk renderer works as expected.
"""

import copy

import numpy as np
from argoverse.utils.polyline_density import get_polyline_length

import tbv.rendering.crosswalk_renderer as crosswalk_renderer


def test_clip_line_segment_to_center() -> None:
    """Ensure a polyline can be trimmed on both sides to satisfy a length requirement (shrunk).

    Shrink from length 4 to length 2.
    """
    # fmt: off
    line_segment = np.array(
        [
            [0,0],
            [0,1],
            [0,2],
            [0,3],
            [0,4]
        ]
    )
    # fmt: on
    clipped_line_segment = crosswalk_renderer.clip_line_segment_to_center(line_segment, clip_len=2)

    # all x values should be 0
    assert np.allclose(clipped_line_segment[:, 0], 0)

    # all y values should be between 1 and 3, to get length 2
    assert np.all(clipped_line_segment[:, 1] > 1)
    assert np.all(clipped_line_segment[:, 1] < 3)


# def test_compute_point_line_distance() -> None:
#     """ """
#     point = np.array([3, 3])
#     line_point1 = np.array([1, 1])
#     line_point2 = np.array([1, 5])
#     dist = crosswalk_renderer.compute_point_line_distance(point, line_point1, line_point2)
#     assert dist == 2

#     point = np.array([3, 3])
#     line_point1 = np.array([0, 4])
#     line_point2 = np.array([4, 0])
#     dist = crosswalk_renderer.compute_point_line_distance(point, line_point1, line_point2)
#     assert np.isclose(dist, np.sqrt(2))


def test_get_polyline_length() -> None:
    """
    Verifies that Argoverse is computing the polyline length correctly.
    """
    polyline = np.array([[0, 0], [2, 0], [4, 0]])
    len = get_polyline_length(polyline)
    assert len == 4

    polyline = np.array([[0, 0], [2, 0]])
    len = get_polyline_length(polyline)
    assert len == 2


def test_compute_line_segment_normal() -> None:
    """Ensure we can compute the normal vector to a 2d line segment.

    Define normal to (1,0) vector as (0,1).
    """
    # fmt: off
    ls = np.array(
        [
            [2., 4.],
            [4., 4.]
        ]
    )
    # fmt: on
    normal = crosswalk_renderer.compute_line_segment_normal(ls)
    # should be unit length
    expected_normal = np.array([0.0, 1.0])
    assert np.allclose(normal, expected_normal)


def test_get_rectangular_region_horizontal_mismatched_len() -> None:
    """
    Ensure get_rectangular_region() returns the expected shape when creating crosswalk,
    for two horizontal lines.

       X -> X  edge 2

    X --------> X    edge 1
    """
    # fmt: off
    # Note: floating point arguments required
    edge1 = np.array(
        [
            [0, 0],
            [10.0, 0],
        ]
    )
    edge2 = np.array(
        [
            [2, 4],
            [4, 4.0],
        ]
    )
    # fmt: on
    v0, v1, v2, v3 = crosswalk_renderer.get_rectangular_region(edge1=copy.deepcopy(edge1), edge2=copy.deepcopy(edge2))

    # new edge1 is along y = 4
    assert np.allclose(v0, np.array([2, 4]))
    assert np.allclose(v1, np.array([4, 4]))

    # new edge2 is along y = 0
    assert np.allclose(v2, np.array([2, 0]))
    assert np.allclose(v3, np.array([4, 0]))

    # now make sure swapped order of arguments works too, should yield identical
    v0, v1, v2, v3 = crosswalk_renderer.get_rectangular_region(edge1=copy.deepcopy(edge2), edge2=copy.deepcopy(edge1))

    # new edge1 is along y = 4 (shorter side)
    assert np.allclose(v0, np.array([2, 4]))
    assert np.allclose(v1, np.array([4, 4]))

    # new edge2 is along y = 0
    assert np.allclose(v2, np.array([2, 0]))
    assert np.allclose(v3, np.array([4, 0]))


def test_get_rectangular_region_vertical_mismatched_len() -> None:
    """
    Ensure get_rectangular_region() returns the expected shape when creating crosswalk,
    for two vertical lines.

    edge 1    
    //\\
     |       edge 2
     |       //\\
     |        |
     |
    """
    # fmt: off
    # Note: floating point arguments required
    edge1 = np.array(
        [
            [0, 0],
            [0, 10.0]
        ]
    )
    edge2 = np.array(
        [
            [4, 2],
            [4, 4.0]
        ]
    )
    # fmt: on
    v0, v1, v2, v3 = crosswalk_renderer.get_rectangular_region(edge1, edge2)

    # new edge1 is along x = 4 (shorter side)
    assert np.allclose(v0, np.array([4, 2]))
    assert np.allclose(v1, np.array([4, 4]))

    # new edge2 is along x = 0
    assert np.allclose(v2, np.array([0, 2]))
    assert np.allclose(v3, np.array([0, 4]))

    # now make sure swapped order of arguments works too, should yield identical
    v0, v1, v2, v3 = crosswalk_renderer.get_rectangular_region(edge1=copy.deepcopy(edge2), edge2=copy.deepcopy(edge1))

    # new edge1 is along x = 4 (shorter side)
    assert np.allclose(v0, np.array([4, 2]))
    assert np.allclose(v1, np.array([4, 4]))

    # new edge2 is along x = 0
    assert np.allclose(v2, np.array([0, 2]))
    assert np.allclose(v3, np.array([0, 4]))


def test_get_rectangular_region_diagonal_same_len() -> None:
    """
    Ensure get_rectangular_region() returns the expected shape when creating crosswalk,
    for two diagonal lines.

         \\
           \\ edge 2
             \\
    \\        -|
      \\
        \\ edge 11
        --|
    """
    # fmt: off
    # Note: floating point arguments required
    edge1 = np.array(
        [
            [0, 2],
            [2, 0.0]
        ]
    )
    edge2 = np.array(
        [
            [4, 6],
            [6, 4.0]
        ]
    )
    # fmt: on
    v0, v1, v2, v3 = crosswalk_renderer.get_rectangular_region(edge1, edge2)

    # edge1 preserved as edge1 (not swapped), since both have the same length. Already rectangular.
    assert np.allclose(v0, np.array([0.0, 2.0]))
    assert np.allclose(v1, np.array([2.0, 0.0]))
    assert np.allclose(v2, np.array([4.0, 6.0]))
    assert np.allclose(v3, np.array([6.0, 4.0]))

    # now make sure swapped order of arguments works too, should swap lines, but keep rectangle, since now first arg
    # is considered longest (equal, so length compare treated as false).
    v0, v1, v2, v3 = crosswalk_renderer.get_rectangular_region(edge1=copy.deepcopy(edge2), edge2=copy.deepcopy(edge1))

    # same as input
    assert np.allclose(v0, np.array([4.0, 6.0]))
    assert np.allclose(v1, np.array([6.0, 4.0]))
    # same as input
    assert np.allclose(v2, np.array([0.0, 2.0]))
    assert np.allclose(v3, np.array([2.0, 0.0]))
