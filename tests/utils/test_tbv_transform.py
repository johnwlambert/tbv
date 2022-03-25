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

import tbv.utils.tbv_transform as tbv_transform

def test_CropBottomSquarePair() -> None:
    """ """
    img = np.arange(10 * 5 * 3).reshape(10, 5, 3).astype(np.uint8)
    label = np.arange(10 * 5).reshape(10, 5).astype(np.uint8)

    transform = tbv_transform.CropBottomSquarePair()
    cropped_img, cropped_label = transform(img, label)
    assert cropped_img.shape == (5, 5, 3)
    assert cropped_label.shape == (5, 5)
