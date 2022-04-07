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
Utilities to apply a median filter to data.
"""

from collections import defaultdict

import numpy as np


def form_aggregate_rgb_img(
    bev_img: np.ndarray, bev_img_pts: np.ndarray, rgb_vals: np.ndarray, method: str = "median"
) -> np.ndarray:
    """

    Note: extremely slow.

    Form 3d tensor, then using median value along each fibre
    or mean value ('mean')
    or last value ('naive')

    Args:
        bev_img: array of shape ()
        bev_img_pts: array of shape ()
        rgb_vals: array of shape ()
        method

    Returns:
        bev_img: array of shape ()
    """
    if method == "naive":
        # use naive strategy (last-indexed val is visible)
        bev_img[bev_img_pts[:, 1], bev_img_pts[:, 0]] = rgb_vals
        return bev_img

    assert method in ["median", "mean"]
    img_h, img_w, _ = bev_img.shape
    repeat_dict = defaultdict(int)

    num_pts = bev_img_pts.shape[0]
    assert bev_img_pts.shape[1] == 2
    assert rgb_vals.shape == (num_pts, 3)
    for (x, y) in bev_img_pts:
        repeat_dict[(x, y)] += 1

    max_count = max(list(repeat_dict.values()))
    print("Most repeated location: ", max_count)

    color_tensor = np.zeros((img_h, img_w, max_count, 3))
    curr_count_dict = defaultdict(int)

    # fill a 4d tensor with the possible values
    for k in range(num_pts):
        x, y = bev_img_pts[k]
        curr_count_idx = curr_count_dict[(x, y)]
        color_tensor[y, x, curr_count_idx] = rgb_vals[k]
        curr_count_dict[(x, y)] += 1

    # loop through all image locations
    # if nonempty, place median into bev_img
    for i in range(img_h):
        for j in range(img_w):
            is_valid_row, is_valid_col = np.where(color_tensor[i, j] != 0)

            if is_valid_row.size == 0:
                continue
            max_row = np.amax(is_valid_row)

            for ch in range(3):
                if method == "median":
                    bev_img[i, j, ch] = np.median(color_tensor[i, j, : max_row + 1, ch])
                elif method == "mean":
                    bev_img[i, j, ch] = int(np.mean(color_tensor[i, j, : max_row + 1, ch]))
    return bev_img


if __name__ == "__main__":

    # test_form_aggregate_rgb_img()
    # test_form_aggregate_rgb_img_partially_populated()
    test_form_aggregate_rgb_img_oom()
