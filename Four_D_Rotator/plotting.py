"""
Matplotlib visualisation helpers for *tesseract_slice*.

Separated from geometry so non-GUI environments can still import core math.

Author: Michael Bagalman
License: MIT
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Michael Bagalman

from __future__ import annotations

from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ._constants import DEFAULT_COLORMAP, TOL
from .geometry import SliceError, slice_tesseract

__all__ = [
    "plot_slice",
    "create_rotation_animation",
]

# ---------------------------------------------------------------------------
# Helper: enforce equal aspect ratio
# ---------------------------------------------------------------------------

def _set_equal_aspect(ax: plt.Axes, vertices: np.ndarray) -> None:
    """Adjust 3D axes so that x, y, z have equal scale."""
    max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2.0
    center = vertices.mean(axis=0)
    for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(c - max_range, c + max_range)


# ---------------------------------------------------------------------------
# Plotting routine
# ---------------------------------------------------------------------------

ColorSpec = Union[str, List[float], np.ndarray]
Edge = Tuple[np.ndarray, np.ndarray]

def plot_slice(
    angles: Dict[str, float],
    *,
    w_fixed: float = 0.0,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
    show_vertices: bool = True,
    vertex_size: int = 60,
    vertex_color: ColorSpec = "red",
    color_by_distance: bool = False,
    colormap: str = DEFAULT_COLORMAP,
    show_edges: bool = True,
    edge_color: ColorSpec = "black",
    edge_width: float = 1.5,
    show_axes: bool = True,
    show_info: bool = True,
    title: Optional[str] = None,
    elev: float = 20,
    azim: float = 45,
    tol: float = TOL,
) -> plt.Figure:
    """Return a matplotlib Figure visualising the 3D slice."""
    verts, edges = slice_tesseract(angles, w_fixed=w_fixed, tol=tol)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()
        ax.clear()

    if show_edges and edges:
        lc = Line3DCollection(edges, colors=edge_color, linewidths=edge_width, alpha=0.8)
        ax.add_collection3d(lc)

    if show_vertices:
        if color_by_distance:
            d = np.linalg.norm(verts, axis=1)
            colors = plt.cm.get_cmap(colormap)(d / d.max())
        else:
            colors = vertex_color
        ax.scatter(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            s=vertex_size,
            c=colors,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
        )

    _set_equal_aspect(ax, verts)

    if show_axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    if title is None:
        title = f"Slice with {len(verts)} vertices"
    ax.set_title(title, pad=20)

    if show_info:
        info_lines = [f"W = {w_fixed:+.3f}", f"Vertices: {len(verts)}"]
        angle_txt = ", ".join(f"{k}={v:+.2f}" for k, v in angles.items()) or "(none)"
        info_lines.append(f"Rot: {angle_txt}")
        ax.text2D(
            0.02,
            0.98,
            "\n".join(info_lines),
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7),
            fontsize=9,
        )

    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Animation helper
# ---------------------------------------------------------------------------

def create_rotation_animation(
    rotation_axis: str,
    *,
    angle_range: Tuple[float, float] = (0.0, 2 * np.pi),
    n_frames: int = 60,
    base_angles: Optional[Dict[str, float]] = None,
    w_fixed: float = 0.0,
    **plot_kwargs,
) -> Generator[plt.Figure, None, None]:
    """Yield successive Figures rotating *rotation_axis* across *angle_range*.

    Yes, it's just a for-loop over frames, but if it makes you feel better,
    consider it a tribute to the Infinite Improbability Drive.
    """

    if rotation_axis not in {"xy", "xz", "yz", "xw", "yw", "zw"}:
        raise ValueError(f"invalid rotation plane '{rotation_axis}'")

    fixed = dict(base_angles or {})
    # The axis must rotate. The axis *always* rotates. (Muad'Dib intensifies.)
    for ang in np.linspace(angle_range[0], angle_range[1], n_frames):
        rot = fixed.copy(); rot[rotation_axis] = ang
        try:
            fig = plot_slice(rot, w_fixed=w_fixed, **plot_kwargs)
            yield fig
        except SliceError:
            continue  # skip frames where plane misses
