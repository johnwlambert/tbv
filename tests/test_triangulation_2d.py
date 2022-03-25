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

import time

import mapbox_earcut as earcut
import matplotlib.pyplot as plt
import numpy as np
import shapely.ops
import trimesh
from argoverse.utils.manhattan_search import compute_point_cloud_bbox
from argoverse.utils.mpl_plotting_utils import plot_lane_segment_patch
from shapely.geometry import Polygon



def test_form_polygon_triangulation_2d() -> None:
    """ """
    # in-order
    polygon = np.array(
        [
            [1,1],
            [1,2],
            [2,3],
            [3,3],
            [3,2],
            [4,2],
            [3,1]
        ])
    triangles = form_polygon_triangulation_2d(polygon)
    gt_triangles = np.array(
        [
            [1., 2., 2., 1., 2., 2.],
            [2., 2., 3., 1., 3., 2.],
            [2., 3., 3., 2., 3., 3.],
            [1., 1., 2., 1., 1., 2.],
            [2., 1., 3., 1., 2., 2.],
            [2., 2., 3., 2., 2., 3.]
        ])
    assert np.allclose(gt_triangles, triangles)

    # try higher resolution
    triangles = form_polygon_triangulation_2d(polygon, 1.0)

    visualize = False
    if not visualize:
        return

    plot_triangulation(triangles, polygon=polygon_pts)



def test_form_polygon_triangulation_2d_lane() -> None:
    """ """
    polygon_pts = np.array(
        [
            [6971.7042871 , 1336.34449988,   58.83628209],
            [6972.67261929, 1334.31773551,   58.90108236],
            [6972.71816971, 1334.10843957,   58.9075466 ],
            [6972.97288364, 1333.58696165,   58.92061441],
            [6973.19547362, 1333.07555888,   58.93432384],
            [6973.27364244, 1332.64976751,   58.94256135],
            [6973.19849815, 1332.27001786,   58.9483057 ],
            [6972.97941657, 1331.91758011,   58.95733129],
            [6972.44071336, 1331.46242834,   58.97675736],
            [6971.52448612, 1330.99605191,   59.01243634],
            [6971.42924255, 1330.93054608,   59.0178402 ],
            [6973.0017454 , 1326.8303181 ,   59.17974106],
            [6975.04010803, 1327.94417462,   59.12891217],
            [6976.40139368, 1329.25760764,   59.11822077],
            [6977.15940018, 1330.74796338,   59.10248287],
            [6977.38799637, 1332.39254184,   59.07992372],
            [6977.16084612, 1334.16868818,   59.0483554 ],
            [6976.55165328, 1336.05381974,   59.00721015],
            [6975.58338239, 1338.08045338,   58.94408053],
            [6971.7042871 , 1336.34449988,   58.83628209]
        ]
    )

    start = time.time()
    triangles = form_polygon_triangulation_2d(
        polygon_pts=polygon_pts,
        downsample_factor=0.1,
        allow_interior_only=True,
    )
    end = time.time()
    duration = end - start
    print(f"Duration: {duration:.2f} sec. for lane boundary polygon.")

    plot_triangulation(triangles, polygon=polygon_pts)





def test_form_polygon_triangulation_2d_rectangle() -> None:
    """ """
    polygon_pts = np.array(
        [
            [7088.60240482, 1211.75775711],
            [7088.60240482, 1411.75775711],
            [6888.60240482, 1411.75775711],
            [6888.60240482, 1211.75775711]
        ]
    )

    start = time.time()
    triangles = form_polygon_triangulation_2d(
        polygon_pts=polygon_pts,
        downsample_factor=0.1,
        allow_interior_only=False
    )
    end = time.time()
    duration = end - start
    print(f"Duration: {duration:.2f} sec. for rectangle")

    plot_triangulation(triangles, polygon=polygon_pts)


def plot_triangulation(triangles: np.ndarray, polygon: np.ndarray) -> None:
    """ """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(polygon[:,0], polygon[:,1], color='r')

    plot_lane_segment_patch(
        polygon_pts=polygon,
        ax=ax,
        color='b' #np.random.rand(3)
    )

    for tri in triangles:
        polygon_pts = tri.reshape(3,2)
        polygon_pts = np.vstack([ polygon_pts, polygon_pts[0] ])
        plot_lane_segment_patch(
            polygon_pts=polygon_pts,
            ax=ax,
            color=np.random.rand(3)
        )
    #plt.scatter(0,0, 10, marker='.', color='r')
    plt.axis("equal")
    plt.show()
    plt.close("all")



def test_subdivide_mesh() -> None:
    """
    (0,2)    (2,2)
      .-------.
      |     / |
      |   /   |
      | /     |
      .-------.
    (0,0)   (2,0)
    """
    triangles = np.array(
        [
            [0,0,2,0,2,2],
            [0,0,2,2,0,2]
        ]
    )
    subdivided_triangles = subdivide_mesh(triangles)

    plot_triangulation(subdivided_triangles, polygon=np.array([[0,0],[2,0],[2,2],[0,2]]))




def test_triangulation() -> None:
    """ """
    points = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    # tri = scipy.spatial.Delaunay(points)
    # import pdb; pdb.set_trace()
    

    pdb.set_trace()


if __name__ == "__main__":

    test_form_polygon_triangulation_2d_lane()
    test_form_polygon_triangulation_2d_rectangle()

    # test_subdivide_mesh()
