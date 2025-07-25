"""
Compute statistics for tesseract slices.

Author: Michael Bagalman
License: MIT

This module collects a few helpful numeric descriptors of a slice,
like edge length metrics and bounding box info. Use this when
you want something slightly more profound than just a pretty picture.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Michael Bagalman

from __future__ import annotations

from typing import Dict
import numpy as np
from scipy.spatial import ConvexHull

from .geometry import slice_tesseract, Edge

__all__ = ["edge_metrics", "analyze_slice"]


# ---------------------------------------------------------------------------
# Edge length statistics
# ---------------------------------------------------------------------------

def edge_metrics(edges: list[Edge]) -> Dict[str, float]:
    """Return basic edge length statistics from the given geometry.

    Parameters
    ----------
    edges : list of (np.ndarray, np.ndarray)
        Each edge is a 2-tuple of 3D coordinates.

    Returns
    -------
    dict with keys: count, min, max, mean, std, median
    """
    lengths = [np.linalg.norm(p - q) for p, q in edges]
    return {
        "count": len(lengths),
        "min": float(np.min(lengths)),
        "max": float(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "median": float(np.median(lengths)),
    }


# ---------------------------------------------------------------------------
# Full analysis summary
# ---------------------------------------------------------------------------

def analyze_slice(angles: Dict[str, float], w_fixed: float = 0.0, tol: float = 1e-6) -> Dict[str, object]:
    """Return centroid, bounding box, and edge statistics for a slice.

    This is the function you call when you want the numbers to speak.

    Parameters
    ----------
    angles : dict
        Dict of 4D rotation angles.
    w_fixed : float
        The fixed W-plane for slicing.
    tol : float
        Numerical tolerance for edge detection.

    Returns
    -------
    dict with keys: centroid, bounding_box, vertices, edge_metrics
    """
    verts, edges = slice_tesseract(angles, w_fixed=w_fixed, tol=tol)
    centroid = np.mean(verts, axis=0)
    mins = np.min(verts, axis=0)
    maxs = np.max(verts, axis=0)
    bounding_box = {
        "x": [mins[0], maxs[0]],
        "y": [mins[1], maxs[1]],
        "z": [mins[2], maxs[2]],
    }

    return {
        "centroid": centroid,
        "bounding_box": bounding_box,
        "vertices": verts,
        "edge_metrics": edge_metrics(edges),
    }
