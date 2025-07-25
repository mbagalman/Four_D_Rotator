"""
Global constants for the *tesseract_slice* package.

Includes common tolerances, valid axis names, and configurable defaults
used across modules.

Author: Michael Bagalman
License: MIT
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Michael Bagalman

# ---------------------------------------------------------------------------
# Numerical tolerances and axis naming conventions
# ---------------------------------------------------------------------------

TOL: float = 1e-8  # Global tolerance for slicing comparisons

# Valid keys for 4D rotation specification
VALID_KEYS: set[str] = {"xy", "xz", "yz", "xw", "yw", "zw"}

# Mapping vertex counts → human-friendly shape names
SHAPE_NAMES: dict[int, str] = {
    4: "tetrahedron",
    6: "triangular prism",
    8: "cube",
    12: "hexagonal prism",
    14: "heptagonal prism",
    16: "hyperoctahedron",
    18: "octagonal prism",
    20: "dodecahedron",
    24: "rhombicuboctahedron",
    32: "truncated cube",
    48: "truncated octahedron",
    64: "tesseract slice",
    96: "snub polychoron?",
    120: "120-cell slice?",
    600: "600-cell slice?",
}

# ---------------------------------------------------------------------------
# Plotting defaults
# ---------------------------------------------------------------------------

DEFAULT_COLORMAP: str = "viridis"  # Moved from plotting.py for consistency
