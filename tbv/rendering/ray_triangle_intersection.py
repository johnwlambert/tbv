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

from typing import Optional, Tuple

import numpy as np


def compute_triangle_plane_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute normal to plane spanned by triangle's 3 vertices
       v2
      /  \
     /    \
    v0 --- v1

    Args:
        v0: triangle vertex 0, ordered counter-clockwise (CCW)
        v1: triangle vertex 1
        v2: triangle vertex 2
    """
    assert all([v.dtype in [np.float32, np.float64] for v in [v0, v1, v2]])
    # compute plane's normal
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    N = np.cross(v0v1, v0v2)
    N /= np.linalg.norm(N)
    return N


def inside_outside_test(N: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, P: np.ndarray) -> bool:
    """
    Compute normal to plane spanned by triangle's 3 vertices
       v2
      /  \
     /    \
    v0 --- v1

    C is a vector perpendicular to triangle's plane 

    Args:
        v0: triangle vertex 0, ordered counter-clockwise (CCW)
        v1: triangle vertex 1
        v2: triangle vertex 2

    Returns:
        boolean indicating...
	"""
    # edge 0
    edge0 = v1 - v0
    vp0 = P - v0
    C = np.cross(edge0, vp0)
    if N.dot(C) < 0:
        return False  # P is on the right side

    # edge 1
    edge1 = v2 - v1
    vp1 = P - v1
    C = np.cross(edge1, vp1)
    if N.dot(C) < 0:
        return False  # P is on the right side

    # edge 2
    edge2 = v0 - v2
    vp2 = P - v2
    C = np.cross(edge2, vp2)
    if N.dot(C) < 0:
        return False  # P is on the right side

    return True


def inside_outside_test_vectorized(
    N: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, Ps: np.ndarray
) -> np.array:
    """
    Args:
        Ps: points on the triangle's plane. Not necessarily within
            the triangle. These came from ray-plane intersection.

    Returns:
        boolean indicating whether plane point falls within triangle
    """
    # edge 0
    edge0 = v1 - v0
    # edge 1
    edge1 = v2 - v1
    # edge 2
    edge2 = v0 - v2

    vp0 = Ps - v0
    vp1 = Ps - v1
    vp2 = Ps - v2

    C0 = np.cross(edge0, vp0)
    C1 = np.cross(edge1, vp1)
    C2 = np.cross(edge2, vp2)

    # if (dot product < 0), then P is on the right side
    return np.logical_and.reduce([C0.dot(N) >= 0, C1.dot(N) >= 0, C2.dot(N) >= 0])


def ray_triangle_intersect(
    origin: np.ndarray, ray_dir: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> Tuple[bool, Optional[np.ndarray]]:
    """t is the distance along the ray from the origin

    Args:
        origin: shape (3,)
        ray_dir: ray direction shape (3,)
        v0: triangle vertex 0, ordered counter-clockwise (CCW)
        v1: triangle vertex 1
        v2: triangle vertex 2

    Returns:
        boolean whether intersection is valid
        P: intersection point if exists, otherwise None
    """
    N = compute_triangle_plane_normal(v0, v1, v2)

    # Step 1: finding P
    # check if ray and plane are parallel ?
    NdotRayDirection = N.dot(ray_dir)
    kEpsilon = 1e-10
    if np.absolute(NdotRayDirection) < kEpsilon:  # almost 0
        return False, None  # they are parallel so they don't intersect !

    # compute d parameter of implicit line equation
    d = N.dot(v0)

    # compute t (equation 3)
    t = (d - N.dot(origin)) / NdotRayDirection
    # check if the triangle is in behind the ray
    if t < 0:
        return False, None  # the triangle is behind

    # compute the intersection point using ray parameterization
    P = origin + t * ray_dir

    is_inside = inside_outside_test(N, v0, v1, v2, P)
    if not is_inside:
        return False, None

    return True, P  # this ray hits the triangle


def ray_triangle_intersect_moller_trombore(
    origin: np.ndarray, ray_dir: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> Tuple[bool, Optional[np.ndarray]]:
    """t is the distance along the ray from the origin

    Args:
        origin: shape (3,)
        ray_dir: ray direction shape (3,)
        v0: triangle vertex 0, ordered counter-clockwise (CCW)
        v1: triangle vertex 1
        v2: triangle vertex 2

    Returns:
        boolean indicating whether intersection is valid
        P: intersection point if exists, otherwise None
    """
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    pvec = np.cross(ray_dir, v0v2)
    det = v0v1.dot(pvec)

    # CULLING
    # if the determinant is negative the triangle is backfacing
    # if the determinant is close to 0, the ray misses the triangle
    kEpsilon = 1e-10
    if det < kEpsilon:
        return False, None

    invDet = 1 / det

    tvec = origin - v0
    u = tvec.dot(pvec) * invDet
    if (u < 0) or (u > 1):
        return False, None

    qvec = np.cross(tvec, v0v1)
    v = ray_dir.dot(qvec) * invDet
    if (v < 0) or (u + v > 1):
        return False, None

    t = v0v2.dot(qvec) * invDet

    P = origin + t * ray_dir
    return True, P


def ray_triangle_intersect_vectorized_moller_trombore(
    origin: np.ndarray, ray_dirs: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Use Cramer's rule and Barycentric coordinates, per
    https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates

    Args:
        ray_dirs: N x 3 for directions of N ray

    Returns:
        valid: array of bool, whether hit or not
        Ps: array of intersection points, otherwise NULL values
    """
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    pvec = np.cross(ray_dirs, v0v2)
    det = pvec.dot(v0v1)

    # CULLING
    # if the determinant is negative the triangle is backfacing
    # if the determinant is close to 0, the ray misses the triangle
    kEpsilon = 1e-10
    # valid = det >= kEpsilon

    invDet = 1 / det

    tvec = origin - v0
    u = pvec.dot(tvec) * invDet

    # valid = np.logical_and.reduce(
    # 	[
    # 		u >= 0,
    # 		u <= 1,
    # 		valid
    # 	])

    qvec = np.cross(tvec, v0v1)
    v = ray_dirs.dot(qvec) * invDet
    valid = np.logical_and.reduce(
        [
            det >= kEpsilon,
            u >= 0,
            u <= 1,
            v >= 0,
            u + v <= 1
            # valid
        ]
    )
    t = v0v2.dot(qvec) * invDet

    # compute the intersection point using ray parameterization
    # see broadcast example below (so last dims match for multiply)
    Ps = origin + (t * ray_dirs.T).T
    return valid, Ps


def ray_triangle_intersect_vectorized(
    origin: np.ndarray, ray_dirs: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """

    Args:
        ray_dirs: N x 3 for directions of N ray

    Returns:
        valid: array of bool, whether hit or not
        Ps: array of intersection points, otherwise NULL values
    """
    N = compute_triangle_plane_normal(v0, v1, v2)

    # Step 1: finding P
    # check if ray and plane are parallel ?
    NdotRayDirections = ray_dirs.dot(N)
    kEpsilon = 1e-10

    # if almost 0, they are parallel so they don't intersect !
    valid = np.absolute(NdotRayDirections) > kEpsilon

    # compute d parameter of implicit line equation
    d = N.dot(v0)

    DENOMINATOR_PADDING = 100
    NdotRayDirections[~valid] += DENOMINATOR_PADDING

    # compute t -- a vector of distances along the ray
    t = (d - N.dot(origin)) / NdotRayDirections

    # compute the intersection point using ray parameterization
    # see broadcast example below (so last dims match for multiply)
    Ps = origin + (t * ray_dirs.T).T

    is_inside = inside_outside_test_vectorized(N, v0, v1, v2, Ps)

    # check if the triangle is in behind the ray
    # if t < 0, then # the triangle is behind
    valid = np.logical_and.reduce([valid, t >= 0, is_inside])

    # if true, then # this ray hits the triangle
    return valid, Ps
