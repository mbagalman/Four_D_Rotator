"""
Handy rotation presets for *tesseract_slice*.

Author: Michael Bagalman
License: MIT

These are the greatest hits — a sampler pack of 4D rotations.
They’re great for demos, debugging, and those moments when you’re too lazy
(“strategic,” I mean) to type your own angle dicts.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Michael Bagalman

import numpy as np
from typing import Dict

AngleDict = Dict[str, float]

__all__ = ["standard_rotations"]


def standard_rotations() -> Dict[str, AngleDict]:
    """Return a dict of name → angle-dict presets suitable for demos.

    A good starter kit when you want to show students what slicing a tesseract
    *actually* looks like instead of staring into the existential abyss of 4D math.
    """

    return {
        "identity": {},
        "xy_45": {"xy": np.pi / 4},
        "xy_90": {"xy": np.pi / 2},
        "xyz_equal": {"xy": np.pi / 6, "xz": np.pi / 6, "yz": np.pi / 6},
        "4d_symmetric": {"xy": np.pi / 4, "xz": np.pi / 4, "xw": np.pi / 4},
        "complex": {
            "xy": 0.6,
            "xz": 0.7,
            "yz": 0.4,
            "xw": 0.9,
            "yw": 1.2,
            "zw": 0.3,
        },
        "rhombic_dodeca": {"xy": 0.5, "xw": 0.5, "yz": 0.3},
        "octahedron": {"xy": np.pi / 3, "xw": np.pi / 4},
        "tetrahedron": {"xy": 0.2, "xz": 0.2, "xw": 1.2, "yw": 0.8},
    }
