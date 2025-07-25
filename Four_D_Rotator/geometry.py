"""
Geometry module: slice a tesseract into 3D using 4D hyperplane rotation.
Author: Michael Bagalman
License: MIT

This is the mathematical heart of the operation. It doesn't care about your
widgets or matplotlib backends — it just wants to rotate a 4D cube and dissect
it like an alien autopsy.

NOTE: This version has been updated to correctly calculate slices by intersecting
the tesseract's edges with the slicing plane, rather than only checking vertices.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Michael Bagalman

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ._constants import TOL, VALID_KEYS

__all__ = ["Edge", "slice_tesseract", "SliceError"]


# ---------------------------------------------------------------------------
# Types and error class
# ---------------------------------------------------------------------------

Edge = Tuple[np.ndarray, np.ndarray]  # Each edge is a pair of 3D points

class SliceError(Exception):
    """Raised when the slicing plane fails to intersect the tesseract,
    or the resulting intersection is degenerate (a point, line, or 2D polygon)."""
    pass


# ---------------------------------------------------------------------------
# Core math utilities
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_tesseract_geometry() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate vertices and edges for a 4D hypercube centered at the origin.
    The hypercube has a side length of 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - A (16, 4) numpy array of vertex coordinates.
        - A (32, 2) numpy array of edge indices.
    """
    verts = np.array(
        [
            [x, y, z, w]
            for x in (-0.5, 0.5)
            for y in (-0.5, 0.5)
            for z in (-0.5, 0.5)
            for w in (-0.5, 0.5)
        ]
    )
    
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            # An edge connects vertices if the distance is 1 (side length)
            if np.isclose(np.linalg.norm(verts[i] - verts[j]), 1.0):
                edges.append((i, j))
    
    return verts, np.array(edges)


@lru_cache(maxsize=128)
def _rotation_matrix(angles: Tuple[Tuple[str, float], ...]) -> np.ndarray:
    """
    Generate a 4D rotation matrix from a list of plane-angle tuples.

    Example input: (("xy", 0.5), ("zw", -0.2))
    This function supports composition of rotations in multiple planes.
    Order matters, so we sort them in slice_tesseract before caching.
    """
    mat = np.eye(4)
    for plane, theta in angles:
        if plane not in VALID_KEYS:
            raise ValueError(f"Invalid rotation plane: {plane}")
        i, j = "xyzw".index(plane[0]), "xyzw".index(plane[1])
        rot = np.eye(4)
        c, s = np.cos(theta), np.sin(theta)
        rot[i, i] = rot[j, j] = c
        rot[i, j] = -s
        rot[j, i] = s
        mat = rot @ mat
    return mat


# ---------------------------------------------------------------------------
# Slicing and projection
# ---------------------------------------------------------------------------

def slice_tesseract(
    angles: Dict[str, float], *, w_fixed: float = 0.0, tol: float = TOL
) -> Tuple[np.ndarray, List[Edge]]:
    """
    Intersect a tesseract with the hyperplane w = w_fixed, apply 4D rotation,
    and return 3D vertices and visible edges.

    This version calculates intersections of the tesseract's edges with the
    slicing plane, providing a robust slicing method that can generate all
    possible convex polyhedral slices.

    Parameters
    ----------
    angles : dict
        Dictionary of plane names (like "xy" or "zw") and rotation angles in radians.
    w_fixed : float
        The hyperplane value to slice at (default: 0.0). A value outside the
        range of roughly [-1, 1] may not intersect the centered hypercube.
    tol : float
        Tolerance for floating point comparisons.

    Returns
    -------
    Tuple[np.ndarray, List[Edge]]
        - A NumPy array of 3D vertex coordinates forming the slice.
        - A list of edges (pairs of 3D points) for the convex hull of the slice.
    
    Raises
    ------
    SliceError
        If the plane does not intersect the tesseract or the intersection is
        degenerate (a point, line, or 2D polygon) and cannot form a 3D shape.
    """
    verts4d_orig, edge_indices = _get_tesseract_geometry()
    
    if not all(k in VALID_KEYS for k in angles.keys()):
        invalid = [k for k in angles.keys() if k not in VALID_KEYS]
        raise ValueError(f"Invalid angle keys provided: {invalid}")

    rot = _rotation_matrix(tuple(sorted(angles.items())))
    rotated_verts = verts4d_orig @ rot.T

    intersection_points = []
    for i, j in edge_indices:
        p1 = rotated_verts[i]
        p2 = rotated_verts[j]
        w1, w2 = p1[3], p2[3]

        # Case 1: Edge crosses the slicing plane.
        if (w1 < w_fixed and w2 > w_fixed) or (w2 < w_fixed and w1 > w_fixed):
            # Linearly interpolate to find the intersection point.
            if np.isclose(w1, w2):
                continue
            t = (w_fixed - w1) / (w2 - w1)
            intersection = p1 + t * (p2 - p1)
            intersection_points.append(intersection[:3])
        # Case 2: One of the vertices lies exactly on the plane.
        elif np.isclose(w1, w_fixed, atol=tol):
            intersection_points.append(p1[:3])
        elif np.isclose(w2, w_fixed, atol=tol):
            intersection_points.append(p2[:3])

    if not intersection_points:
        raise SliceError("Slice plane does not intersect the tesseract.")

    # Remove duplicate points using rounding to a set precision.
    # This handles cases where multiple edges meet at a vertex on the slicing plane.
    unique_pts_tuples = {tuple(np.round(p, 8)) for p in intersection_points}
    slice_pts = np.array([list(p) for p in unique_pts_tuples])

    if slice_pts.shape[0] < 4:
        raise SliceError(f"Slice resulted in only {slice_pts.shape[0]} unique vertices, not enough for a 3D shape.")

    try:
        # Use ConvexHull to find the boundary of the intersection points.
        # 'QJ' option tells Qhull to jitter points for precision issues.
        hull = ConvexHull(slice_pts, qhull_options="QJ")
    except QhullError as e:
        # This can happen if all points are coplanar, forming a 2D polygon.
        raise SliceError(f"Convex hull failed, the slice is likely 2D or degenerate. Qhull error: {e}")

    edge_set = set()
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            a, b = sorted((simplex[i], simplex[(i + 1) % len(simplex)]))
            edge_set.add((a, b))

    # The vertices of the slice are the points that form the convex hull.
    final_verts = hull.points 
    edges: List[Edge] = [(final_verts[i], final_verts[j]) for i, j in edge_set]
    
    return final_verts, edges
