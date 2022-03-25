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

import tbv.utils.frustum_utils as frustum_utils


def test_get_frustum_side_normals() -> None:
    """Ensure we can compute normals to left and right clipping planes accurately.

    Scenario uses a frustum with a fov of 90 degrees (PI/2 radians). Suppose
    in the egovehicle frame, the camera looks directly down the +x direction,
    from the origin.
    """
    fov_theta = np.pi / 2
    import pdb; pdb.set_trace()
    l_normal, r_normal = frustum_utils.get_frustum_side_normals(fov_theta)

    # normals for left and right clipping planes point into the frustum
    gt_l_normal = np.array([1.0, -1.0]) / np.sqrt(2)
    assert np.allclose(l_normal, gt_l_normal)

    gt_r_normal = np.array([1.0, 1.0]) / np.sqrt(2)
    assert np.allclose(r_normal, gt_r_normal)

    # frustum with width 53 degrees
    # when we point in (x,y)=(2,1) direction (narrower frustum)
    fov_theta = np.deg2rad(53.13010235415598)
    l_normal, r_normal = frustum_utils.get_frustum_side_normals(fov_theta)

    gt_l_normal = np.array([1.0, -2.0]) / np.linalg.norm(np.array([2, 1]))
    assert np.allclose(l_normal, gt_l_normal)

    gt_r_normal = np.array([1.0, 2.0]) / np.linalg.norm(np.array([2, 1]))
    assert np.allclose(r_normal, gt_r_normal)



# def test_get_frustum_parameters() -> None:
#     """ """
#     log_ids = ["allison-arkansas-wdc-new-bollards", "new-bike-lane-v5"]

#     data_dir = "/home/jlambert/mcd_extraction_output_dir_2020_07_17/logs"
#     dl = SimpleArgoverseTrackingDataLoader(data_dir=data_dir, labels_dir=data_dir)

#     for log_id in log_ids:
#         log_calib_data = dl.get_log_calibration_data(log_id)

#         # for each camera frustum
#         for camera_name in RING_CAMERA_LIST:
#             logger.info(f"On camera {camera_name}")
#             camera_config = get_calibration_config(log_calib_data, camera_name)
#             yaw, fov = frustum_utils.get_frustum_parameters(camera_config)

#             if camera_name == "ring_front_center":
#                 assert np.isclose(yaw, -0.01, atol=1)
#                 assert np.isclose(fov, 85.21, atol=1)
#             if camera_name == "ring_front_left":
#                 assert np.isclose(yaw, 45.11, atol=1)
#                 assert np.isclose(fov, 101.08, atol=1)
#             if camera_name == "ring_side_left":
#                 assert np.isclose(yaw, 99.23, atol=1)
#                 assert np.isclose(fov, 101.10, atol=1)
#             if camera_name == "ring_rear_left":
#                 assert np.isclose(yaw, 153.21, atol=1)
#                 assert np.isclose(fov, 101.04, atol=1)
#             if camera_name == "ring_rear_right":
#                 assert np.isclose(yaw, -152.99, atol=1)
#                 assert np.isclose(fov, 101.14, atol=1)
#             if camera_name == "ring_side_right":
#                 assert np.isclose(yaw, -98.97, atol=1)
#                 assert np.isclose(fov, 101.14, atol=1)
#             if camera_name == "ring_front_right":
#                 assert np.isclose(yaw, -44.99, atol=1)
#                 assert np.isclose(fov, 101.10, atol=1)
