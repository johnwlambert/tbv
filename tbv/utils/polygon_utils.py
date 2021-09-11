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

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LinearRing, Point, Polygon


def polygon_pt_dist(polygon: Polygon, pt: Point, show_plot: bool = False) -> float:
    """Returns polygon to point distance
    
    Args:
        polygon: as a shapely Polygon
        pt: as a shapely Point
        show_plot: boolean indicating whether to visualize the objects using Matplotlib.

    Returns:
        dist: float representing distance ...
    """
    pol_ext = LinearRing(polygon.exterior.coords)
    # distance along the ring to a point nearest the other object.
    d_geodesic = pol_ext.project(pt)
    # return a point at the specified distance along the ring
    nearest_p = pol_ext.interpolate(d_geodesic)
    diff = np.array(nearest_p.coords) - np.array(pt)
    dist = np.linalg.norm(diff)

    if show_plot:
        pt_coords = np.array(pt.coords).squeeze()
        nearest_p_coords = np.array(nearest_p.coords).squeeze()

        plt.scatter(pt_coords[0], pt_coords[1], 10, color="k")
        plt.scatter(nearest_p_coords[0], nearest_p_coords[1], 10, color="r")

        plt.plot(*polygon.exterior.xy, color="b")
        plt.show()

    return dist
