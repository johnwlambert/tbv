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

import tbv.utils.lidar_io as lidar_io


def test_load_tbv_sweep_invalid_attrib_spec() -> None:
    """Invalid attribute specification shoud lead to returning `None` for sweep."""

    pt_fpath = "dummy_fname.pt"
    # "a" and "b" are invalid, meaningless channel identifiers
    attrib_spec = "xyzab"
    sweep_xyzil = lidar_io.load_tbv_sweep(pt_fpath, attrib_spec)
    assert sweep_xyzil is None