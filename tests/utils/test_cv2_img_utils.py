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

import tbv.utils.cv2_img_utils as cv2_img_utils


def test_draw_polygon_cv2_smokescreen() -> None:
    """
    Test ability to fill a nonconvex polygon.
    We don't verify the rendered values since this requires
    scanline rendering computation to find the polygon's
    exact boundaries on a rasterized grid.
    """
    UINT8_MAX = 255
    img_w = 40
    img_h = 20
    for dtype in [np.uint8, np.float32]:

        # (u,v) points: Numpy array of shape (N,2)
        pentagon_pts = np.array([[1, 0], [2, 2], [0, 4], [-2, 2], [-1, 0]])
        # move the pentagon origin to (10,20) so in image center
        pentagon_pts[:, 0] += int(img_w / 2)
        pentagon_pts[:, 1] += int(img_h / 2)
        # img: Numpy array of shape (M,N,3)
        img = np.ones((img_h, img_w, 3), dtype=dtype) * UINT8_MAX
        # color: Numpy array of shape (3,)
        color = np.array([255.0, 0.0, 0.0])

        img_w_polygon = cv2_img_utils.draw_polygon_cv2(pentagon_pts, img.copy(), color)
        import matplotlib.pyplot as plt; plt.imshow(img_w_polygon); plt.show()
        assert isinstance(img_w_polygon, np.ndarray)
        assert img_w_polygon.shape == img.shape
        assert img_w_polygon.dtype == dtype
