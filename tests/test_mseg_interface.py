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

import tbv.utils.mseg_interface as mseg_interface


def test_get_mseg_label_map_fpath_from_image_info() -> None:
    """ """
    label_maps_dir = "/path/to/label/maps"
    log_id = "abc__2020_06_01"
    camera_name = "ring_rear_left"
    img_fname_stem = "ring_rear_left_9999"

    label_map_fpath = mseg_interface.get_mseg_label_map_fpath_from_image_info(
        label_maps_dir, log_id, camera_name, img_fname_stem
    )
    assert (
        label_map_fpath
        == "/path/to/label/maps/mseg-3m-480_abc__2020_06_01_ring_rear_left_universal_ss/358/gray/ring_rear_left_9999.png"
    )


def test_filter_by_semantic_classes() -> None:
    """Ensure that pixels corresponding to non-ground surfaces are marked properly.

    Consider 4 pixel locations in a 3x2 px image (height 3 px, width 2 px).
    """
    u_classnames = names_utils.get_universal_class_names()
    road_uid = u_classnames.index("road")
    sidewalk_uid = u_classnames.index("sidewalk_pavement")
    car_uid = u_classnames.index("car")
    # fmt: off
    label_map = np.array(
        [
            [car_uid, car_uid],
            [road_uid, car_uid],
            [sidewalk_uid, car_uid]
        ]
    )
    # fmt: on
    uv = np.array(
        [
            # x,y
            [1, 0],  # car
            [0, 1],  # road
            [1, 1],  # car
            [0, 2],  # sidewalk
        ]
    )
    logicals = mseg_interface.filter_by_semantic_classes(label_map, uv, render_road_only=False)
    gt_logicals = np.array([False, True, False, True])
    assert np.allclose(logicals, gt_logicals)
