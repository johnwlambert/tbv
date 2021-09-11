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

from typing import List

import numpy as np


def hstack_imgs(img_list: List[np.ndarray]) -> np.ndarray:
    """Horizontally stack images of the same height"""
    img_h, img_w, ch = img_list[0].shape
    assert ch == 3

    # height and number of channels must match
    assert all(img.shape[0] == img_h for img in img_list)
    assert all(img.shape[2] == ch for img in img_list)

    num_imgs = len(img_list)

    all_widths = [img.shape[1] for img in img_list]
    hstack_img = np.zeros((img_h, sum(all_widths), 3), dtype=np.uint8)

    running_w = 0
    for i, img in enumerate(img_list):
        h, w, _ = img.shape
        start = running_w
        end = start + w
        hstack_img[:, start:end, :] = img
        running_w += w

    return hstack_img
