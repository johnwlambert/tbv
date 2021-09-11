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

import logging
from typing import List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.interpolate import compute_midpoint_line
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Point, Polygon

import tbv.utils.infinity_norm_utils as infinity_norm_utils
from tbv.rendering.map_rendering_classes import LocalVectorMap, LocalPedCrossing
from tbv.utils.polygon_utils import polygon_pt_dist

WPT_INFTY_NORM_INTERP_NUM = 50


"""
The map perturbation routine calls `insert_synthetic_crosswalk()` to add a crosswalk into the
local vector map at a plausible location.

The rest of the functions in this file support that single function call.

Future TODO: could convert tuple to numpy arrays to typed Crosswalk object.
"""

# maximum allowed IoU between a real crosswalk and a synthetically generated crosswalk, for acceptance.
MAX_INTER_XWALK_IOU = 0.05


def get_lane_polygon_union_2d(lane_polygons: List[np.ndarray], visualize: bool = False) -> Polygon:
    """Get 2d union only"""
    lane_polygon_union = Polygon(lane_polygons[0][:, :2])
    for lane_polygon in lane_polygons:
        lane_polygon_union = Polygon(lane_polygon[:, :2]).union(lane_polygon_union)
        if visualize:
            plt.plot(lane_polygon[:, 0], lane_polygon[:, 1], color="m")

    return lane_polygon_union


def extend_linestring(ls: LineString) -> LineString:
    """Extend a line segment, representing as a LineString, equally in both directions, to increase its length.

    For example, a line segment of length 8 from (4,4) <-> (4,12) would become (4,4-800) <-> (4,12+800).

    Args:
        ls: line segment between two endpoints, expressed as a LineString

    Returns:
        line segment extended by 100 times its length, in each direction. Line's normal is preserved.
    """
    polyline = np.array(ls.coords)

    # get vector representation of A->B, given endpoints A and B.
    diff_vec = np.diff(polyline, axis=0).squeeze()

    polyline[0] -= diff_vec * 100
    polyline[1] += diff_vec * 100
    return LineString(polyline)


def build_random_crosswalk_from_edge(
    right_polyline: np.ndarray, cw_width: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Given one edge along the principal axis of a crosswalk, create a second edge to bound its area.

    We parameterize each crosswalk by its left and right edge, along its principal axis.
    Other option would be to do ls.parallel_offset(cw_width, 'right'),
    but we use parallel_offset to the left.

    Given A->B (right polyline), we would generate C->D as follows (left polyline)
    C ---> D

    A ---> B

    Args:
        right_polyline: array of shape (2,2) representing one edge of a crosswalk.
        cw_width: desired width of crosswalk, in meters (optional). If not provided, then the crosswalk
            width is sampled from a truncated normal distribution. This determinism is needed to make
            reasonable unit tests, but is not used when generating training data.

    Returns:
        right_polyline: array of shape (2,2) representing right polyline endpoints (identical to input)
        left_polyline: array of shape (2,2) representing left polyline endpoints
            This 2nd edge is to the left of the first edge.
    """
    left_pt = right_polyline[0]
    right_pt = right_polyline[1]
    ls = LineString([left_pt, right_pt])

    # sample crosswalk width if not explicitly specified.
    if cw_width is None:
        # truncated normal distribution
        cw_width = np.random.normal(loc=3.5)
        cw_width = np.clip(cw_width, a_min=2, a_max=4.0)

    left = ls.parallel_offset(cw_width, "left")

    left_polyline = np.array(left.boundary)
    return (right_polyline, left_polyline)


def find_xwalk_iou_2d(xwalk_1: Tuple[np.ndarray, np.ndarray], xwalk_2: Tuple[np.ndarray, np.ndarray]) -> float:
    """Find the IoU between two crosswalks (in the 2d plane).

    Args:
        xwalk_1: crosswalk #1, represented as tuple of left and right polylines. each polyline is an array
            of shape (2,2)
        xwalk_2: crosswalk #2, with same format as crosswalk #1.

    Returns:
        float in range [0,1] representing intersection-over-union.
    """
    xing1_verts = np.vstack(xwalk_1)[:, :2]
    xing2_verts = np.vstack(xwalk_2)[:, :2]

    # TODO: remove the ConvexHull calls
    hull1_idxs = ConvexHull(xing1_verts).vertices
    hull2_idxs = ConvexHull(xing2_verts).vertices

    poly1 = Polygon(xing1_verts[hull1_idxs])
    poly2 = Polygon(xing2_verts[hull2_idxs])

    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union


def find_1d_closest_idx_to_origin(values: np.ndarray, side_to_search: str) -> int:
    """Given a 1d array of scalar values, find the value closest to zero on either the negative or positive side.

    Args:
        values: 1d array of shape (N,) containing positive and negative values
        side_to_search: closest positive or closest negative value

    Returns:
        integer-valued index representing position of value that satisfies the query.
    """
    # TODO: handle edge cases when no valid index exists.
    all_idxs = np.arange(values.size)
    if side_to_search == "positive":
        valid_idxs = all_idxs[values >= 0]

    elif side_to_search == "negative":
        valid_idxs = all_idxs[values < 0]

    valid_dists_to_origin = np.square(values[valid_idxs])
    smallest_idx = np.argmin(valid_dists_to_origin)

    return valid_idxs[smallest_idx]


def get_rotmat2d(theta_deg: float) -> np.ndarray:
    """Converts angle in degrees to 2d rotation matrix.

    Args:
        theta_deg: rotation angle, measured in degrees.

    Returns:
        R: array of shape (2,2) representing a 2d rotation matrix.
    """
    # convert to radians
    theta_rad = np.deg2rad(theta_deg)
    sin = np.sin(theta_rad)
    cos = np.cos(theta_rad)
    R = np.array([[cos, -sin], [sin, cos]])
    return R


class Line2D(NamedTuple):
    """Representation of a 2d line defined by ax + by + c = 0"""

    a: float
    b: float
    c: float

    def get_normal_vec(self) -> np.ndarray:
        """ """
        return np.array([self.a, self.b])

    def get_point_signed_dists_to_line(self, pts: np.ndarray) -> np.ndarray:
        """Compute signed point-to-line distance for each input point.

        See
        - https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        - https://brilliant.org/wiki/dot-product-distance-between-point-and-a-line/

        Args:
            pts: array of shape (N,2) representing N two-dimensional points.

        Returns:
            dists_to_line: array of shape (N,) representing signed distances from the line to each point.
        """
        pts = pts.astype(np.float32)
        dists_to_line = pts.dot(self.get_normal_vec()) + self.c
        dists_to_line /= np.linalg.norm([self.a, self.b])
        return dists_to_line

    def get_closest_pt(self, pts: np.ndarray, direction: str) -> np.ndarray:
        """
        Args:
            pts:
            direction: either 'positive' or 'negative'

        Returns:
            point ....
        """
        dists_to_line = self.get_point_signed_dists_to_line(pts)

        idx_closest = find_1d_closest_idx_to_origin(dists_to_line, direction)
        return pts[idx_closest]

    def get_extended_normal_linestring(self, center_waypt: np.ndarray) -> LineString:
        """Get a line segment (represented by a LineString) centered at a point and extending
        in both directions about that point, parallel to this Line2D's (self's) normal.

        Useful when we now the normal to a lane centerline for example, and want to extend a
        crosswalk in this normal direction across the lane centerline.

        Args:
            center_waypt: center of desired line segment

        Returns:
            ls: generated line segment. Left endpoint is in the direction of the normal. Right
                endpoint is in the opposite direction of the normal.
        """
        if center_waypt.size == 3:
            raise ValueError("Center waypoint argument must be 2d, not 3d, for Line2D.get_extended_normal_linestring")

        left_pt = center_waypt + self.get_normal_vec()
        right_pt = center_waypt - self.get_normal_vec()

        # draw the line between them
        ls = LineString([left_pt, right_pt])
        # increase line segment's length by 100x in each direction
        ls = extend_linestring(ls)
        return ls


def get_polyline_normal_2d(polyline_pts: np.ndarray, waypt_idx: int) -> Line2D:
    """Get normal that crosses perpendicular over polyline at specified waypoint index.

    Using central difference for derivative approximation.

    Args:
        polyline_pts: array of shape (N,2) consisting of N waypoints.
        waypt_idx: index of waypoint at which to compute polyline normal. Cannot be 0 or N-1.

    Returns:
        Line2D object, parameterized by the two-dimensional line's normal vector.
    """
    if polyline_pts.shape[1] not in [2, 3]:
        raise RuntimeError("Wrong shape input to get_polyline_normal_2d()")
    if polyline_pts.shape[1] == 3:
        # just do the math in 2d
        polyline_pts = polyline_pts[:, :2]

    # must have at least two points
    if polyline_pts.shape[0] <= 2:
        raise RuntimeError("Need at least 2 points as input to get_polyline_normal_2d()")

    # cannot be the zero'th or last point, to do central difference.
    if waypt_idx <= 0 or waypt_idx >= (polyline_pts.shape[0] - 1):
        raise RuntimeError("Waypoint index is out of bounds; cannot be negative.")

    polyline_tangent = polyline_pts[waypt_idx + 1] - polyline_pts[waypt_idx - 1]

    R = get_rotmat2d(90)

    # compute normal!
    normal_vec = R.dot(polyline_tangent)
    normal_vec /= np.linalg.norm(normal_vec)

    # normal vector n = [a,b]
    # ax + by + c = 0, so c = -ax + -by
    x, y = polyline_pts[waypt_idx]
    a, b = normal_vec
    c = -np.array([x, y]).dot(np.array([a, b]))

    return Line2D(a, b, c)


def insert_synthetic_crosswalk(
    lvm: LocalVectorMap, ego_center: np.ndarray, window_sz: float, visualize: bool = False
) -> LocalVectorMap:
    """Insert a synthetically-generated crosswalk into the local vector map, which could be rendered
    in the BEV or ego-view.

    -----From our paper's appendix:----
    Our main observations from studying mapped data are that crosswalks are generally located near intersections,
    are orthogonal to lane segment tangents, and have little to no area overlap with other crosswalks. Accordingly,
    we first sample a random lane segment which will be spanned by the generated, synthetic crosswalk. We perform
    this random sampling from a biased but normalized probability distribution; lane segments within intersections
    achieve 4.5x the weight of non-intersection lane segments. In order to determine the orientation of the
    synthesized crosswalk's principal axis, we compute the normal to the centerline of the sampled lane segment at
    a randomly sampled waypoint. This waypoint is sampled from 50 waypoints that we interpolate along the centerline.
    We ensure that the sampled waypoint is not within the outermost 1/8 of pixels along any border of the rendered
    map image (i.e. within 15 m according to l-infinity norm from the egovehicle). This measure is to allow some
    perturbation of the random crop for data augmentation, without losing visibility of the changed entity.

    Next, in order to determine how many total lane segments the crosswalk must cross in order to span the entire
    road, we must determine the road extent. We approximate it as the union of all nearby lane segment polygons.
    The line representing the principal axis of the crosswalk may intersect with this road polygon in more than
    two locations, since it is often non-convex. We choose the shortest possible length segment that spans the
    road polygon to be valid, and thus find the closest two intersections to the sampled centerline waypoint.
    We randomly sample a crosswalk width $w$ in meters from a normal distribution w ~ N(3.5,1), but clip to the
    range w in [2,4] meters afterwards, in accordance to our empirical observations of the real-world distribution.

    If the rendered synthetic crosswalk has overlap with any other real crosswalk above a threshold of IoU=0.05,
    we continue to sample until we succeed. The crosswalk is rendered as a rectangle, bounded between two long
    edges both extending along the principal axis of the crosswalk. We use alternating parallel strips of white
    and gray to color the object. Crosswalks are deleted by simply not rendering them in the rasterized image.

    Args:
        lvm:
        ego_center: array of shape ()
        window_sz:
        visualize: boolean indicating whether to use matplotlib to visualize the scene and generated elements.

    Returns:
        lvm: local vector map, with a synthetically-generated crosswalk inserted.
    """
    lane_polygons = [vls.polygon_boundary for vls in lvm.nearby_lane_segment_dict.values()]
    lane_polygon_union = get_lane_polygon_union_2d(lane_polygons, visualize)

    iou_w_other_crosswalk = 1.0
    while iou_w_other_crosswalk > MAX_INTER_XWALK_IOU:

        sampled_lane_id = lvm.sample_lane_id_biased_to_intersections()

        rand_right_lane_boundary = lvm.nearby_lane_segment_dict[sampled_lane_id].right_lane_boundary.xyz
        rand_left_lane_boundary = lvm.nearby_lane_segment_dict[sampled_lane_id].left_lane_boundary.xyz

        # sample random waypoint along any lane boundary
        # dont choose first or last, for finite differences
        # note that high index is exclusive for randint.
        waypt_idx = np.random.randint(low=1, high=WPT_INFTY_NORM_INTERP_NUM - 1)

        # we dont need to interpolate in 3d, since insertion reasoning is in 2d.
        centerline_pts, _ = compute_midpoint_line(
            left_ln_bnds=rand_left_lane_boundary[:,:2],
            right_ln_bnds=rand_right_lane_boundary[:,:2],
            num_interp_pts=WPT_INFTY_NORM_INTERP_NUM,
        )

        if not infinity_norm_utils.has_pts_in_infty_norm_radius(
            centerline_pts[waypt_idx - 1 : waypt_idx + 2], ego_center, window_sz - 5
        ):
            print("Sampled waypoint outside infinity norm radius, skipping")
            continue

        assert infinity_norm_utils.has_pts_in_infty_norm_radius(
            centerline_pts[waypt_idx - 1 : waypt_idx + 2], ego_center, window_sz - 5
        )
        # get 2d line
        lane_normal_line = get_polyline_normal_2d(centerline_pts, waypt_idx=waypt_idx)
        normal_ls = lane_normal_line.get_extended_normal_linestring(center_waypt=centerline_pts[waypt_idx])

        inter_ls = lane_polygon_union.intersection(normal_ls)
        try:
            inter_ls = np.array(inter_ls.coords)
            left_pt = inter_ls[0]
            right_pt = inter_ls[1]
            if visualize:
                plt.scatter(left_pt[0], left_pt[1], 10, color="r")
                plt.scatter(right_pt[0], right_pt[1], 10, color="r")

        except:
            candidate_pts = []

            linestrings = list(inter_ls.geoms)
            colors = ["c", "y", "g", "k", "m"]
            # find the closest possible point in each direction
            for i, inter_ls in enumerate(linestrings):
                color = colors[i]
                inter_ls = np.array(inter_ls.coords)
                left_pt = inter_ls[0]
                right_pt = inter_ls[1]
                if visualize:
                    plt.scatter(left_pt[0], left_pt[1], 10, color=color)
                    plt.scatter(right_pt[0], right_pt[1], 10, color=color)

                POLYGON_DST_THRESH = 0.1
                if polygon_pt_dist(lane_polygon_union, Point(left_pt)) < POLYGON_DST_THRESH:
                    candidate_pts += [left_pt]
                if polygon_pt_dist(lane_polygon_union, Point(right_pt)) < POLYGON_DST_THRESH:
                    candidate_pts += [right_pt]

            # find the closest point with positive angle to normal
            # find the closest point with negative angle to normal
            candidate_pts = np.array(candidate_pts)
            left_pt = lane_normal_line.get_closest_pt(candidate_pts, direction="positive")
            right_pt = lane_normal_line.get_closest_pt(candidate_pts, direction="negative")

            if visualize:
                plt.scatter(left_pt[0], left_pt[1], 30, color="r")
                plt.scatter(right_pt[0], right_pt[1], 30, color="r")

        if visualize:
            plt.scatter(centerline_pts[:, 0], centerline_pts[:, 1], 10, color="k")
            plt.plot(*lane_polygon_union.exterior.xy, color="b")
            plt.show()

        right_polyline = np.vstack([left_pt.reshape(1, 2), right_pt.reshape(1, 2)])
        proposed_xing_edges = build_random_crosswalk_from_edge(right_polyline)

        # extend it to the boundary of the drivable area, loop over true crosswalks
        ious = [find_xwalk_iou_2d(t_xwalk.get_edges(), proposed_xing_edges) for t_xwalk in lvm.ped_crossing_edges]
        print("Crosswalk IoUs: ", np.round(ious,2))
        # break early if there were no other crosswalks present,
        # or if it overlaps with all other crosswalk polygon by less than 5%
        if len(ious) == 0 or np.array(ious).max() < MAX_INTER_XWALK_IOU:

            # lift the 2D crosswalk (two 2D line segments) to 3D using the map.
            proposed_xing_edges = list(proposed_xing_edges)
            for i, edge in enumerate(proposed_xing_edges):
                proposed_xing_edges[i] = lvm.avm.append_height_to_2d_city_pt_cloud(pt_cloud_xy=edge)

            lvm.ped_crossing_edges += [LocalPedCrossing(edge1=proposed_xing_edges[0], edge2=proposed_xing_edges[1])]

            lvm.changed_points = np.vstack([proposed_xing_edges[0], proposed_xing_edges[1]])
            logging.info("Shape lvm.changed_points: " + str(lvm.changed_points.shape))
            break

    return lvm
