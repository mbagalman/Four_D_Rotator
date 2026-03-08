import unittest
import inspect

import numpy as np

from Four_D_Rotator._constants import TOL
from Four_D_Rotator.analysis import analyze_slice
from Four_D_Rotator.geometry import SliceError, slice_tesseract
from Four_D_Rotator.presets import standard_rotations


def _canonical_vertices(vertices, decimals=8):
    rounded = [tuple(np.round(v, decimals)) for v in vertices]
    return tuple(sorted(rounded))


class GeometryTests(unittest.TestCase):
    def test_analyze_slice_default_tol_uses_shared_constant(self):
        sig = inspect.signature(analyze_slice)
        self.assertEqual(sig.parameters["tol"].default, TOL)

    def test_no_intersection_raises_slice_error(self):
        with self.assertRaises(SliceError):
            slice_tesseract({}, w_fixed=2.0)

    def test_known_vertex_counts_for_presets(self):
        presets = standard_rotations()
        expected_counts = {
            "identity": 8,
            "4d_symmetric": 12,
            "octahedron": 12,
        }
        for name, expected in expected_counts.items():
            vertices, _ = slice_tesseract(presets[name])
            self.assertEqual(len(vertices), expected, msg=f"Unexpected vertex count for {name}")

    def test_rotation_order_is_respected(self):
        angles_a = {"xy": 0.6, "xw": 0.9}
        angles_b = {"xw": 0.9, "xy": 0.6}

        verts_a, _ = slice_tesseract(angles_a)
        verts_b, _ = slice_tesseract(angles_b)

        self.assertNotEqual(_canonical_vertices(verts_a), _canonical_vertices(verts_b))


if __name__ == "__main__":
    unittest.main()
