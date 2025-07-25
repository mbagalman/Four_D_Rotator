"""
JSON export utilities and configuration dataclass for *tesseract_slice*.

Author: Michael Bagalman
License: MIT
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Michael Bagalman

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from . import __version__
from .analysis import analyze_slice
from .geometry import slice_tesseract
from ._constants import TOL

Edge = Tuple[np.ndarray, np.ndarray]
EdgeJSON = Tuple[List[float], List[float]]

__all__ = ["export_to_json", "SliceConfig"]


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_to_json(
    vertices: np.ndarray,
    edges: List[Edge],
    angles: Dict[str, float],
    *,
    w_fixed: float = 0.0,
    filename: Union[str, Path] = "slice_data.json",
    include_analysis: bool = True,
) -> None:
    """Serialize slice data and metadata to JSON.

    Edges are auto-converted from ndarray pairs to nested Python lists.
    """
    edges_as_lists = [[p.tolist(), q.tolist()] for p, q in edges]

    data: Dict[str, Any] = {
        "metadata": {
            "generator": "tesseract_slice",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": __version__,
        },
        "parameters": {"angles": angles, "w_fixed": w_fixed},
        "geometry": {
            "vertices": vertices.tolist(),
            "edges": edges_as_lists,
            "vertex_count": int(len(vertices)),
            "edge_count": int(len(edges)),
        },
    }

    if include_analysis:
        analysis = analyze_slice(angles, w_fixed=w_fixed)
        analysis.pop("vertices", None)  # not serializable
        analysis["centroid"] = analysis["centroid"].tolist()
        bb = analysis["bounding_box"]
        analysis["bounding_box"] = {k: v.tolist() for k, v in bb.items()}
        data["analysis"] = analysis

    Path(filename).write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SliceConfig:
    """Container for slice parameters with convenience methods.

    Useful for saving/loading reusable configurations.
    """
    angles: Dict[str, float] = field(default_factory=dict)
    w_fixed: float = 0.0
    tolerance: float = TOL

    def slice(self):
        return slice_tesseract(self.angles, w_fixed=self.w_fixed, tol=self.tolerance)

    def analyze(self):
        return analyze_slice(self.angles, w_fixed=self.w_fixed, tol=self.tolerance)

    def save(self, path: Union[str, Path]):
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]):
        return cls(**json.loads(Path(path).read_text()))

    def __repr__(self):
        angle_str = ", ".join(f"{k}={v:+.3f}" for k, v in self.angles.items()) or "(none)"
        return f"SliceConfig({angle_str}; w={self.w_fixed:+.3f})"
