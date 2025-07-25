"""
Geometry module: slice a tesseract into 3D using 4D hyperplane rotation.
Author: Michael Bagalman
License: MIT

This is the mathematical heart of the operation. It doesn't care about your
widgets or matplotlib backends — it just wants to rotate a 4D cube and dissect
it like an alien autopsy.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Michael Bagalman

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from ._constants import TOL, VALID_KEYS

__all__ = ["Edge", "slice_tesseract", "SliceError"]


# ---------------------------------------------------------------------------
# Types and error class
# ---------------------------------------------------------------------------

Edge = Tuple[np.ndarray, np.ndarray]  # Each edge is a pair of 3D points

class SliceError(Exception):
    """Raised when the slicing plane fails to intersect the tesseract."""
    pass


# ---------------------------------------------------------------------------
# Core math utilities
# ---------------------------------------------------------------------------

def _make_tesseract_vertices() -> np.ndarray:
    """Generate the 16 corner points of a 4D hypercube (a.k.a. tesseract)."""
    return np.array(
        [[x, y, z, w] for x in (0, 1) for y in (0, 1) for z in (0, 1) for w in (0, 1)]
    )


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

    Parameters:
    - angles: Dictionary of plane names (like "xy" or "zw") and rotation angles in radians.
    - w_fixed: The hyperplane value to slice at (default: 0.0).
    - tol: Tolerance for how close to w_fixed a point must be (default: 1e-5).

    Returns:
    - A NumPy array of 3D vertex coordinates.
    - A list of edges (pairs of 3D points) suitable for plotting.
    """
    verts4d = _make_tesseract_vertices()
    rot = _rotation_matrix(tuple(sorted(angles.items())))
    rotated = verts4d @ rot.T

    # Select points close to the slicing plane
    mask = np.abs(rotated[:, 3] - w_fixed) <= tol
    slice_pts = rotated[mask, :3]
    if len(slice_pts) < 4:
        raise SliceError("Slice did not intersect tesseract in a meaningful way.")

    # Use convex hull to get surface edges (triangles), then dedupe into line segments
    hull = ConvexHull(slice_pts)
    edge_indices = set()
    for simplex in hull.simplices:
        for i in range(3):
            a, b = sorted((simplex[i], simplex[(i + 1) % 3]))
            edge_indices.add((a, b))

    edges: List[Edge] = [(slice_pts[i], slice_pts[j]) for i, j in edge_indices]
    return slice_pts, edges
