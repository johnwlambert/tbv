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

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
from mseg.utils.mask_utils import highlight_binary_mask

import tbv.utils.z1_egovehicle_mask_utils as z1_mask_utils


TEST_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "test_data"


def test_get_z1_ring_front_center_mask() -> None:
    """
    allison-arkansas-wdc-new-bollards/ring_front_center
    new-bike-lane-v5/ring_front_center
    pit-liberty-baum-centre-extended-cnst-line-painting/ring_front_center
    """
    img_fnames = [
        "ring_front_center_315973595849927222.jpg",
        "ring_front_center_315969975349927216.jpg",
        "ring_front_center_315971448799927211.jpg",
        "ring_front_center_315970521649927217.jpg",
    ]
    img_dir = TEST_DATA_ROOT / "egovehicle_mask_imgs"
    for img_fname in img_fnames:
        img_fpath = Path(img_dir) / img_fname
        img = imageio.imread(img_fpath)
        mask = z1_mask_utils.get_z1_ring_front_center_mask()

        assert mask.shape == (2048, 1550)
        assert np.isclose(mask.mean(), 0.118, atol=1e-3)

        mask = highlight_binary_mask(mask.astype(np.uint8), img)
        # plt.imshow(mask)
        # plt.show()


def test_get_z1_ring_rear_right_mask() -> None:
    """
    Images come from:
            crittendon-queenschapel-stop-line-diagonal-repaint/ring_rear_right
            allison-arkansas-wdc-new-bollards/ring_rear_right
            pit-liberty-baum-centre-extended-cnst-line-painting/ring_rear_right
    """
    img_fnames = [
        "ring_rear_right_315969976742441189.jpg",
        "ring_rear_right_315970522142441190.jpg",
    ]
    img_dir = f"{TEST_DATA_ROOT}/egovehicle_mask_imgs"
    for img_fname in img_fnames:
        img_fpath = Path(img_dir) / img_fname
        img = imageio.imread(img_fpath)
        mask = z1_mask_utils.get_z1_ring_rear_right_mask()

        assert mask.shape == (775, 1024)
        assert np.isclose(mask.mean(), 0.0898, atol=1e-4)

        mask = highlight_binary_mask(mask.astype(np.uint8), img)
        # plt.imshow(mask)
        # plt.show()


def test_get_z1_ring_rear_left_mask() -> None:
    """
    new-bike-lane-v5/ring_rear_left
    16th-girard-stop-line-repainted-subtle/ring_rear_left
    webster-13thplace-unmapped-speed-bump/ring_rear_left
    """
    img_dir = f"{TEST_DATA_ROOT}/egovehicle_mask_imgs"
    img_fnames = [
        "ring_rear_left_315968681457428270.jpg",
        "ring_rear_left_315969419357428267.jpg",
        "ring_rear_left_315971447957428275.jpg",
    ]
    for img_fname in img_fnames:
        img_fpath = Path(img_dir) / img_fname
        img = imageio.imread(img_fpath)
        mask = z1_mask_utils.get_z1_ring_rear_left_mask()

        assert mask.shape == (775, 1024)
        assert np.isclose(mask.mean(), 0.0827, atol=1e-4)

        mask = highlight_binary_mask(mask.astype(np.uint8), img)
        # plt.imshow(mask)
        # plt.show()


if __name__ == "__main__":
    """ """
    test_get_z1_ring_front_center_mask()
    test_get_z1_ring_rear_left_mask()
    test_get_z1_ring_rear_right_mask()
