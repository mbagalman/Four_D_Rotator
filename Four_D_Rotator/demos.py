"""
Demo helpers: static gallery & interactive sliders.

Author: Michael Bagalman
License: MIT

This module is what happens when you mix education, 4D geometry,
and a sprinkle of user interaction. Great for classes, talks,
or showing off during office hours.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Michael Bagalman

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .plotting import plot_slice
from .presets import standard_rotations
from .geometry import slice_tesseract, SliceError
from ._constants import SHAPE_NAMES

__all__ = ["demo_slices", "interactive_demo"]


# ---------------------------------------------------------------------------
# Static gallery
# ---------------------------------------------------------------------------

def demo_slices(*, figsize: Tuple[float, float] = (15, 10), save_path: Optional[str] = None) -> None:
    """Display a grid of sample tesseract slices for common rotations.

    You know what students love? A wall of mysterious 3D shapes with no
    explanation. But you’re better than that. This makes the weirdness visual.
    """
    presets = list(standard_rotations().items())[:6]
    fig, axes = plt.subplots(2, 3, figsize=figsize, subplot_kw={"projection": "3d"})

    for ax, (name, ang) in zip(axes.flat, presets):
        try:
            # Delegate drawing to central plot function
            plot_slice(ang, ax=ax, show_info=False, color_by_distance=True, vertex_size=40, edge_width=1.2)
            verts, _ = slice_tesseract(ang)
            shape = SHAPE_NAMES.get(len(verts), "unknown")
            ax.set_title(f"{name}\n{shape} ({len(verts)})", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        except SliceError:
            ax.text(0.5, 0.5, 0.5, "No slice", ha="center", va="center")
            ax.set_title(f"{name}\n(no intersection)")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

    plt.suptitle("Tesseract Slice Gallery", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Interactive Jupyter demo
# ---------------------------------------------------------------------------

def interactive_demo() -> None:  # pragma: no cover
    """Fire up a slider interface in Jupyter to rotate the tesseract live.

    Requires `ipywidgets`. If you're not in a Jupyter notebook, this won't do much
    except make you question your life choices.
    """
    try:
        from ipywidgets import FloatSlider, interact
        from IPython.display import display
    except ImportError:
        print("ipywidgets & IPython are required for the interactive demo.")
        return

    sliders = {
        k: FloatSlider(value=0.0, min=-np.pi, max=np.pi, step=0.05, description=k.upper(), continuous_update=False)
        for k in ["xy", "xz", "yz", "xw", "yw", "zw"]
    }

    def _update(**kwargs):
        ang = {k: v for k, v in kwargs.items() if abs(v) > 0.01} or {"xy": 0.0}
        plot_slice(ang, color_by_distance=True)
        plt.show()

    print("Move sliders to explore 4‑D rotations:")
    display(interact(_update, **sliders))
