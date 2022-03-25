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

import tbv.rendering.bev_vector_map_rendering_utils as bev_vector_map_rendering_utils

def test_draw_dashed_polyline():
    """ """
    im_h = 20
    im_w = 20
    thickness = 1
    color = [255, 0, 0]

    # # horizontal, but uniform
    # line_segments_arr = np.array(
    # 	[
    # 		[0,5],
    # 		[12,5],
    # 	])
    # image = np.zeros((im_h,im_w,3))
    # bev_vector_map_rendering_utils.draw_dashed_polyline(
    # 	line_segments_arr,
    # 	image,
    # 	color,
    # 	im_h,
    # 	im_w,
    # 	thickness,
    # 	dash_interval=2,
    # )
    # plt.imshow(image)
    # plt.show()

    # # horizontal, and non-uniform
    # image = np.zeros((im_h,im_w,3))
    # line_segments_arr = np.array(
    # 	[
    # 		[0,5],
    # 		[3,5],
    # 		[4,5],
    # 		[5,5],
    # 		[6,5],
    # 		[8,5],
    # 		[11,5],
    # 		[12,5],
    # 	])
    # bev_vector_map_rendering_utils.draw_dashed_polyline(
    # 	line_segments_arr,
    # 	image,
    # 	color,
    # 	im_h,
    # 	im_w,
    # 	thickness,
    # 	dash_interval=2,
    # )
    # plt.imshow(image)
    # plt.show()

    # polyline has corner
    line_segments_arr = np.array([[1, 4], [5, 4], [5, 9], [5, 19]])
    image = np.zeros((im_h, im_w, 3))
    bev_vector_map_rendering_utils.draw_dashed_polyline(
        line_segments_arr,
        image,
        color,
        im_h,
        im_w,
        thickness,
        dash_interval=2,
    )
    plt.imshow(image)
    plt.show()




def test_draw_polygon_cv2_smokescreen() -> None:
    """
    Test ability to fill a nonconvex polygon.
    We don't verify the rendered values since this requires
    scanline rendering computation to find the polygon's
    exact boundaries on a rasterized grid.
    """
    UINT8_MAX = 255
    img_w = 6
    img_h = 6
    dtype = np.uint8

    # (x,y) points: Numpy array of shape (N,2), not (u,v) but (v,u)
    # pentagon_pts = np.array([[1, 0], [2, 2], [0, 4], [-2, 2], [-1, 0]])
    pentagon_pts = np.array([[1, 0], [2, 2], [0, 4], [-2, 2], [-1, 0]])
    # move the pentagon origin to (10,20) so in image center
    # img: Numpy array of shape (M,N,3)
    img = np.ones((img_h, img_w, 3), dtype=dtype) * UINT8_MAX
    # color: Numpy array of shape (3,)
    color = np.array([255.0, 0.0, 0.0])

    img_w_polygon = draw_polygon_cv2(pentagon_pts, img.copy(), color)
    img_w_polygon = np.flipud(img_w_polygon)

    # right side of a pentagon shown here as 0s
    # background is 255
    clipped_pentagon_mask = np.array(
        [
            [255, 255, 255, 255, 255, 255],
            [0, 255, 255, 255, 255, 255],
            [0, 0, 255, 255, 255, 255],
            [0, 0, 0, 255, 255, 255],
            [0, 0, 255, 255, 255, 255],
            [0, 0, 255, 255, 255, 255],
        ],
        dtype=np.uint8,
    )
    assert np.allclose(clipped_pentagon_mask, img_w_polygon[:, :, 1])
    assert np.allclose(clipped_pentagon_mask, img_w_polygon[:, :, 2])


def test_render_bev_local_map_elements() -> None:
    """ """
    # ego_center = np.array([2842.55, 8491.44])
    ego_center = np.array([2922.03, 8490.114])
    resolution = 0.02
    range_m = 20
    city_id = 72513
    log_id = "wvWtwTIZ3doB1xqX0j21vr0Qbaxcaoet__2020-08-03-Z1F0062"
    maps_storage_dir = "/home/jlambert/mcd_extraction_output_dir_q85_v12_2020_08_15_02_02_50/maps"
    map_dir = f"{maps_storage_dir}/city_{city_id}"
    pdb.set_trace()
    bev_vector_map_rendering_utils.render_bev_local_map_elements(log_id, map_dir, resolution, copy.deepcopy(ego_center), window_sz=range_m)



def debug_plotting() -> None:
    """ """
    lvm = LocalVectorMap(None, ego_center=np.zeros(2), range_m=20)
    lvm.nearby_lane_segment_dict = {
        88935970: LaneSegment(
            id=88935970,
            right_lane_boundary=np.array([[1830.0, -1459.30090405], [1922.99915009, -1495.28594133]]),
            right_mark_type=LaneMarkType.NONE,
            r_neighbor_id=None,
            left_lane_boundary=np.array([[1830.0, -1455.76687273], [1924.0522072, -1492.43157502]]),
            left_mark_type=LaneMarkType.DOUBLE_SOLID_YELLOW,
            l_neighbor_id=88935983,
            predecessors=None,
            successors=[88935987],
            lane_type="AUTONOMOUS",
            polygon_boundary=np.array(
                [
                    [1830.0, -1459.30090405],
                    [1922.99915009, -1495.28594133],
                    [1924.0522072, -1492.43157502],
                    [1830.0, -1455.76687273],
                    [1830.0, -1459.30090405],
                ]
            ),
            is_intersection=False,
            render_l_bound=True,
            render_r_bound=True,
        ),
        88935978: LaneSegment(
            id=88935978,
            right_lane_boundary=np.array(
                [
                    [1950.0, -1498.54042048],
                    [1949.96804369, -1498.52782007],
                    [1943.25821936, -1496.11200925],
                    [1933.70498074, -1492.13422152],
                    [1925.53471492, -1489.06901958],
                ]
            ),
            right_mark_type=LaneMarkType.NONE,
            r_neighbor_id=None,
            left_ln_boundary=np.array(
                [[1950.0, -1502.5437216], [1949.96702489, -1502.53092365], [1924.0522072, -1492.43157502]]
            ),
            left_mark_type=LaneMarkType.DOUBLE_SOLID_YELLOW,
            l_neighbor_id=88935987,
            predecessors=None,
            successors=[88935983],
            lane_type="AUTONOMOUS",
            polygon_boundary=np.array(
                [
                    [1950.0, -1498.54042048],
                    [1949.96804369, -1498.52782007],
                    [1943.25821936, -1496.11200925],
                    [1933.70498074, -1492.13422152],
                    [1925.53471492, -1489.06901958],
                    [1924.0522072, -1492.43157502],
                    [1949.96702489, -1502.53092365],
                    [1950.0, -1502.5437216],
                    [1950.0, -1498.54042048],
                ]
            ),
            is_intersection=False,
            render_l_bound=True,
            render_r_bound=True,
        ),
        88935983: LaneSegment(
            id=88935983,
            right_lane_boundary=np.array(
                [
                    [1925.53471492, -1489.06901958],
                    [1924.16305084, -1488.68417679],
                    [1923.98968451, -1488.62777415],
                    [1923.41021992, -1488.4392615],
                    [1921.25300117, -1487.59655189],
                    [1900.41040655, -1479.12482647],
                    [1895.3539855, -1477.21547849],
                    [1894.50014648, -1476.85732195],
                    [1893.92708884, -1476.54100554],
                    [1893.48916556, -1476.1739302],
                    [1888.74699163, -1472.20264482],
                    [1876.64575502, -1467.55348351],
                    [1870.94773249, -1465.47654434],
                    [1859.20102434, -1460.96434829],
                    [1853.08362284, -1458.53230209],
                    [1832.17616362, -1450.46452545],
                    [1832.03530097, -1450.41332582],
                    [1830.0, -1449.61160952],
                ]
            ),
            right_mark_type=LaneMarkType.NONE,
            r_neighbor_id=None,
            left_lane_boundary=np.array([[1924.0522072, -1492.43157502], [1830.0, -1455.76687273]]),
            left_mark_type=LaneMarkType.DOUBLE_SOLID_YELLOW,
            l_neighbor_id=88935970,
            predecessors=None,
            successors=[88937297],
            lane_type="AUTONOMOUS",
            polygon_boundary=np.array(
                [
                    [1925.53471492, -1489.06901958],
                    [1924.16305084, -1488.68417679],
                    [1923.98968451, -1488.62777415],
                    [1923.41021992, -1488.4392615],
                    [1921.25300117, -1487.59655189],
                    [1900.41040655, -1479.12482647],
                    [1895.3539855, -1477.21547849],
                    [1894.50014648, -1476.85732195],
                    [1893.92708884, -1476.54100554],
                    [1893.48916556, -1476.1739302],
                    [1888.74699163, -1472.20264482],
                    [1876.64575502, -1467.55348351],
                    [1870.94773249, -1465.47654434],
                    [1859.20102434, -1460.96434829],
                    [1853.08362284, -1458.53230209],
                    [1832.17616362, -1450.46452545],
                    [1832.03530097, -1450.41332582],
                    [1830.0, -1449.61160952],
                    [1830.0, -1455.76687273],
                    [1924.0522072, -1492.43157502],
                    [1925.53471492, -1489.06901958],
                ]
            ),
            is_intersection=False,
            render_l_bound=True,
            render_r_bound=True,
        ),
        88935987: LaneSegment(
            id=88935987,
            right_lane_boundary=np.array(
                [[1922.99915009, -1495.28594133], [1949.96620301, -1505.81411982], [1950.0, -1505.82774717]]
            ),
            right_mark_type=LaneMarkType.NONE,
            r_neighbor_id=None,
            left_lane_boundary=np.array(
                [[1924.0522072, -1492.43157502], [1949.96702489, -1502.53092365], [1950.0, -1502.5437216]]
            ),
            left_mark_type=LaneMarkType.DOUBLE_SOLID_YELLOW,
            l_neighbor_id=88935978,
            predecessors=None,
            successors=[88936295],
            lane_type="AUTONOMOUS",
            polygon_boundary=np.array(
                [
                    [1922.99915009, -1495.28594133],
                    [1949.96620301, -1505.81411982],
                    [1950.0, -1505.82774717],
                    [1950.0, -1502.5437216],
                    [1949.96702489, -1502.53092365],
                    [1924.0522072, -1492.43157502],
                    [1922.99915009, -1495.28594133],
                ]
            ),
            is_intersection=False,
            render_l_bound=True,
            render_r_bound=True,
        ),
        88936295: LaneSegment(
            id=88936295,
            right_lane_boundary=np.array(
                [[1950.0, -1505.82774717], [1961.63832439, -1510.52046047], [1973.37863127, -1515.05629778]]
            ),
            right_mark_type=LaneMarkType.NONE,
            r_neighbor_id=None,
            left_lane_boundary=np.array([[1950.0, -1502.5437216], [1974.47728953, -1512.04358208]]),
            left_mark_type=LaneMarkType.DOUBLE_SOLID_YELLOW,
            l_neighbor_id=88936506,
            predecessors=None,
            successors=[88936485],
            lane_type="AUTONOMOUS",
            polygon_boundary=np.array(
                [
                    [1950.0, -1505.82774717],
                    [1961.63832439, -1510.52046047],
                    [1973.37863127, -1515.05629778],
                    [1974.47728953, -1512.04358208],
                    [1950.0, -1502.5437216],
                    [1950.0, -1505.82774717],
                ]
            ),
            is_intersection=False,
            render_l_bound=True,
            render_r_bound=True,
        ),
        88936506: LaneSegment(
            id=88936506,
            right_lane_boundary=np.array([[1975.70571649, -1508.67621235], [1950.0, -1498.54042048]]),
            right_mark_type=LaneMarkType.NONE,
            r_neighbor_id=None,
            left_lane_boundary=np.array([[1974.47728953, -1512.04358208], [1950.0, -1502.5437216]]),
            left_mark_type=LaneMarkType.DOUBLE_SOLID_YELLOW,
            l_neighbor_id=88936295,
            predecessors=None,
            successors=[88935978],
            lane_type="AUTONOMOUS",
            polygon_boundary=np.array(
                [
                    [1975.70571649, -1508.67621235],
                    [1950.0, -1498.54042048],
                    [1950.0, -1502.5437216],
                    [1974.47728953, -1512.04358208],
                    [1975.70571649, -1508.67621235],
                ]
            ),
            is_intersection=False,
            render_l_bound=True,
            render_r_bound=True,
        ),
    }

    lvm.nearby_lane_segment_dict = {
        88935970: LaneSegment(
            id=88935970,
            right_lane_boundary=np.array([[1830.0, -1459.30090405], [1922.99915009, -1495.28594133]]),
            right_mark_type=LaneMarkType.NONE,
            r_neighbor_id=None,
            left_lane_boundary=np.array([[1830.0, -1455.76687273], [1924.0522072, -1492.43157502]]),
            left_mark_type=LaneMarkType.DOUBLE_SOLID_YELLOW,
            l_neighbor_id=88935983,
            predecessors=None,
            successors=[88935987],
            lane_type="AUTONOMOUS",
            polygon_boundary=np.array(
                [
                    [1830.0, -1459.30090405],
                    [1922.99915009, -1495.28594133],
                    [1924.0522072, -1492.43157502],
                    [1830.0, -1455.76687273],
                    [1830.0, -1459.30090405],
                ]
            ),
            is_intersection=False,
            render_l_bound=True,
            render_r_bound=True,
        ),
        88935978: LaneSegment(
            id=88935978,
            right_lane_boundary=np.array(
                [
                    [1950.0, -1498.54042048],
                    [1949.96804369, -1498.52782007],
                    [1943.25821936, -1496.11200925],
                    [1933.70498074, -1492.13422152],
                    [1925.53471492, -1489.06901958],
                ]
            ),
            right_mark_type=LaneMarkType.NONE,
            r_neighbor_id=None,
            left_lane_boundary=np.array(
                [[1950.0, -1502.5437216], [1949.96702489, -1502.53092365], [1924.0522072, -1492.43157502]]
            ),
            left_mark_type=LaneMarkType.DOUBLE_SOLID_YELLOW,
            l_neighbor_id=88935987,
            predecessors=None,
            successors=[88935983],
            lane_type="AUTONOMOUS",
            polygon_boundary=np.array(
                [
                    [1950.0, -1498.54042048],
                    [1949.96804369, -1498.52782007],
                    [1943.25821936, -1496.11200925],
                    [1933.70498074, -1492.13422152],
                    [1925.53471492, -1489.06901958],
                    [1924.0522072, -1492.43157502],
                    [1949.96702489, -1502.53092365],
                    [1950.0, -1502.5437216],
                    [1950.0, -1498.54042048],
                ]
            ),
            is_intersection=False,
            render_l_bound=True,
            render_r_bound=True,
        ),
        88935983: LaneSegment(
            id=88935983,
            right_lane_boundary=np.array(
                [
                    [1925.53471492, -1489.06901958],
                    [1924.16305084, -1488.68417679],
                    [1923.98968451, -1488.62777415],
                    [1923.41021992, -1488.4392615],
                    [1921.25300117, -1487.59655189],
                    [1900.41040655, -1479.12482647],
                    [1895.3539855, -1477.21547849],
                    [1894.50014648, -1476.85732195],
                    [1893.92708884, -1476.54100554],
                    [1893.48916556, -1476.1739302],
                    [1888.74699163, -1472.20264482],
                    [1876.64575502, -1467.55348351],
                    [1870.94773249, -1465.47654434],
                    [1859.20102434, -1460.96434829],
                    [1853.08362284, -1458.53230209],
                    [1832.17616362, -1450.46452545],
                    [1832.03530097, -1450.41332582],
                    [1830.0, -1449.61160952],
                ]
            ),
            right_mark_type="NONE",
            r_neighbor_id=None,
            left_lane_boundary=np.array([[1924.0522072, -1492.43157502], [1830.0, -1455.76687273]]),
            left_mark_type=LaneMarkType.DOUBLE_SOLID_YELLOW,
            l_neighbor_id=88935970,
            predecessors=None,
            successors=[88937297],
            lane_type="AUTONOMOUS",
            polygon_boundary=np.array(
                [
                    [1925.53471492, -1489.06901958],
                    [1924.16305084, -1488.68417679],
                    [1923.98968451, -1488.62777415],
                    [1923.41021992, -1488.4392615],
                    [1921.25300117, -1487.59655189],
                    [1900.41040655, -1479.12482647],
                    [1895.3539855, -1477.21547849],
                    [1894.50014648, -1476.85732195],
                    [1893.92708884, -1476.54100554],
                    [1893.48916556, -1476.1739302],
                    [1888.74699163, -1472.20264482],
                    [1876.64575502, -1467.55348351],
                    [1870.94773249, -1465.47654434],
                    [1859.20102434, -1460.96434829],
                    [1853.08362284, -1458.53230209],
                    [1832.17616362, -1450.46452545],
                    [1832.03530097, -1450.41332582],
                    [1830.0, -1449.61160952],
                    [1830.0, -1455.76687273],
                    [1924.0522072, -1492.43157502],
                    [1925.53471492, -1489.06901958],
                ]
            ),
            is_intersection=False,
            render_l_bound=True,
            render_r_bound=True,
        ),
        88935987: LaneSegment(
            id=88935987,
            right_lane_boundary=np.array(
                [[1922.99915009, -1495.28594133], [1949.96620301, -1505.81411982], [1950.0, -1505.82774717]]
            ),
            right_mark_type=LaneMarkType.NONE,
            r_neighbor_id=None,
            left_lane_boundary=np.array(
                [[1924.0522072, -1492.43157502], [1949.96702489, -1502.53092365], [1950.0, -1502.5437216]]
            ),
            left_mark_type=LaneMarkType.DOUBLE_SOLID_YELLOW,
            l_neighbor_id=88935978,
            predecessors=None,
            successors=[88936295],
            lane_type="AUTONOMOUS",
            polygon_boundary=np.array(
                [
                    [1922.99915009, -1495.28594133],
                    [1949.96620301, -1505.81411982],
                    [1950.0, -1505.82774717],
                    [1950.0, -1502.5437216],
                    [1949.96702489, -1502.53092365],
                    [1924.0522072, -1492.43157502],
                    [1922.99915009, -1495.28594133],
                ]
            ),
            is_intersection=False,
            render_l_bound=True,
            render_r_bound=True,
        ),
        88936295: LaneSegment(
            id=88936295,
            right_lane_boundary=np.array(
                [[1950.0, -1505.82774717], [1961.63832439, -1510.52046047], [1973.37863127, -1515.05629778]]
            ),
            right_mark_type=LaneMarkType.NONE,
            r_neighbor_id=None,
            left_lane_boundary=np.array([[1950.0, -1502.5437216], [1974.47728953, -1512.04358208]]),
            left_mark_type=LaneMarkType.SOLID_YELLOW,
            l_neighbor_id=88936506,
            predecessors=None,
            successors=[88936485],
            lane_type="AUTONOMOUS",
            polygon_boundary=np.array(
                [
                    [1950.0, -1505.82774717],
                    [1961.63832439, -1510.52046047],
                    [1973.37863127, -1515.05629778],
                    [1974.47728953, -1512.04358208],
                    [1950.0, -1502.5437216],
                    [1950.0, -1505.82774717],
                ]
            ),
            is_intersection=False,
            render_l_bound=True,
            render_r_bound=True,
        ),
        88936506: LaneSegment(
            id=88936506,
            right_lane_boundary=np.array([[1975.70571649, -1508.67621235], [1950.0, -1498.54042048]]),
            right_mark_type=LaneMarkType.SOLID_YELLOW,
            r_neighbor_id=None,
            left_lane_boundary=np.array([[1974.47728953, -1512.04358208], [1950.0, -1502.5437216]]),
            left_mark_type=LaneMarkType.DOUBLE_SOLID_YELLOW,
            l_neighbor_id=88936295,
            predecessors=None,
            successors=[88935978],
            lane_type="AUTONOMOUS",
            polygon_boundary=np.array(
                [
                    [1975.70571649, -1508.67621235],
                    [1950.0, -1498.54042048],
                    [1950.0, -1502.5437216],
                    [1974.47728953, -1512.04358208],
                    [1975.70571649, -1508.67621235],
                ]
            ),
            is_intersection=False,
            render_l_bound=True,
            render_r_bound=True,
        ),
    }
    bev_img = np.zeros((2000, 2000, 3), dtype=np.uint8)
    resolution = 0.02
    line_width = 6

    image_SE2_city = SE2(rotation=np.array([[1.0, 0.0], [0.0, 1.0]]), translation=np.array([-1916, 1516]))

    for _, vls in lvm.nearby_lane_segment_dict.items():
        if vls.render_r_bound:
            # bev_img = np.zeros((2000,2000,3), dtype=np.uint8)
            bev_img = render_lane_boundary(bev_img, vls, "right", resolution, image_SE2_city, line_width)
            plt.imshow(bev_img[:, :, ::-1])
            plt.show()
        if vls.render_l_bound:
            # bev_img = np.zeros((2000,2000,3), dtype=np.uint8)
            bev_img = render_lane_boundary(bev_img, vls, "left", resolution, image_SE2_city, line_width)
            plt.imshow(bev_img[:, :, ::-1])
            plt.show()

    plt.imshow(bev_img[:, :, ::-1])
    plt.show()


def test_bev_render_window() -> None:
    """ """
    log_id = "dummy_log"
    window_center = np.array([10, 0])

    polylines = [
        np.array([[7, 3], [11, -1]]),
        np.array([[10, 3], [14, -1]]),
    ]
    polyline_types = ["DOUBLE_DASH_WHITE", "DASHED_YELLOW"]

    bev_img = bev_render_window(
        log_id,
        window_center,
        polylines,
        polyline_types,
        polygon_boundaries=[],
        ped_crossing_edges=[],
        resolution=0.5,  # 0.5 m /px
        range_m=4,
        line_width=1,
    )
    assert bev_img.sum() == 149740





