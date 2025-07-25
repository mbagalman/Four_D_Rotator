"""
Tesseract Slicing Toolkit  
Author: Michael Bagalman  
License: MIT  

This module exposes the core public API for slicing and visualizing 4D tesseracts.  
It’s what you actually want to import—everything else is elbow grease and topology.

Functions:
- slice_tesseract: Rotate and slice a 4D cube into glorious 3D.
- plot_slice: Matplotlib-based rendering of the sliced projection.
- demo_slices: Gallery of classic slices (curated, like fine cheese).
- interactive_demo: Sliders! Sliders everywhere.
- export_to_obj / export_to_json: Save your weird math art.
- SliceConfig: A dataclass so you don’t pass 9 args to a function.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Michael Bagalman
# This module defaults to friendliness unless otherwise rotated.

__version__ = "1.0.0"

from .geometry import slice_tesseract, SliceError
from .plotting import plot_slice
from .demos import demo_slices, interactive_demo
from .io_obj import export_to_obj
from .io_json import export_to_json, SliceConfig

__all__ = [
    "slice_tesseract",
    "plot_slice",
    "demo_slices",
    "interactive_demo",
    "export_to_obj",
    "export_to_json",
    "SliceConfig",
    "SliceError",
]

# If this module seems unnaturally large... you’re probably viewing it from the wrong dimension.
