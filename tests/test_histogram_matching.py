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
Unit tests on histogram matching utilities.
"""

import copy

import numpy as np

import tbv.utils.histogram_matching as histogram_matching_utils


def test_get_correspondences() -> None:
    """Try to find correspondences under very tight distance bound (2 centimeters) between two point clouds.

    Only 2 of 4 queries should have a match.
    """
    # fmt: off
    pts_a = np.array(
        [
            [3, 3],
            [1.0, 1.0],
            [9, 5],
            [4.0, 4.0]
        ]
    )
    # fmt: on

    pts_b = np.array(
        [
            [5, 9],  # not a match with any, because exceeds distance bound by a lot.
            [4.02, 4.01],  # not a match, because exceeds distance bound by a little.
            [1.01, 1.0],  # matches with A1 -> (a=1,b=2) is pair
            [3, 3],  # matches with A0 -> (a=0,b=3) is pair
        ]
    )

    pts_a_idxs, pts_b_idxs = histogram_matching_utils.get_correspondences(pts_a, pts_b, max_dist=0.02)

    expected_matches = [(1, 2), (0, 3)]
    for i, (a_idx, b_idx) in enumerate(zip(pts_a_idxs, pts_b_idxs)):
        assert (a_idx, b_idx) == expected_matches[i]

    # (1,2) is one pair and (0,3) is the other match pair.
    assert np.allclose(pts_a_idxs, np.array([1, 0]))
    assert np.allclose(pts_b_idxs, np.array([2, 3]))


def test_fit_and_apply_linear_model() -> None:
    """Ensure that a linear model y=ax+b can be fit correctly, with a=0.5 and b=0.

    Also verify truncation to valid uint8 range.
    """
    # target distribution has half the intensity as source distribution
    s_values = np.array([0, 2])  # x
    t_values = np.array([0, 1])  # y
    source_to_convert = np.array([-2, 0, 2, 4])

    converted_s_values = histogram_matching_utils.fit_and_apply_linear_model(
        s_values, t_values, source_to_convert, dampening_factor=1.0, verbose=True
    )
    # should be [-1,0,1,2] but after clipping to [0,255], we get the following:
    gt_converted_s_values = np.array([0.0, 0.0, 1.0, 2.0])
    assert np.allclose(converted_s_values, gt_converted_s_values)


def test_match_lighting() -> None:
    """Smokescreen -- make sure we can recover the identity function via affine fit in histogram matching.

    Fit y=ax + b, where a=1 and b=0.
    """
    # fmt: off
    source_vals = np.array(
        [
            [10,20,30],
            [20,40,60]
        ]
    ).astype(np.uint8)
    # target is identical to source
    target_vals = np.array(
        [
            [10,20,30],
            [20,40,60]
        ]
    ).astype(np.uint8)
    # map a few extra data points now
    all_source_vals = np.array(
        [
            [10,20,30],
            [20,40,60],
            [40,80,120],
            [80,160,240]
        ]
    ).astype(np.uint8)

    expected_all_source_vals_adjusted = copy.deepcopy(all_source_vals)

    # fmt: on
    all_source_vals_adjusted = histogram_matching_utils.match_lighting(
        source_vals, target_vals, all_source_vals, dampening_factor=1.0
    )

    # fitted affine mapping should be an identity function
    assert np.allclose(expected_all_source_vals_adjusted, all_source_vals_adjusted)
