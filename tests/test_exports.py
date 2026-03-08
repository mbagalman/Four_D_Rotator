import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Four_D_Rotator.geometry import slice_tesseract
from Four_D_Rotator.io_json import export_to_json
from Four_D_Rotator.io_obj import export_to_obj


class ExportTests(unittest.TestCase):
    def test_export_to_json_writes_expected_structure(self):
        angles = {"xy": 0.3, "xw": 0.8}
        vertices, edges = slice_tesseract(angles)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "slice.json"
            export_to_json(vertices, edges, angles, filename=out)

            self.assertTrue(out.exists())
            data = json.loads(out.read_text())

            self.assertIn("metadata", data)
            self.assertIn("parameters", data)
            self.assertIn("geometry", data)
            self.assertIn("analysis", data)
            self.assertEqual(data["geometry"]["vertex_count"], len(vertices))
            self.assertEqual(data["geometry"]["edge_count"], len(edges))

    def test_export_to_json_uses_provided_geometry_for_analysis(self):
        angles_for_vertices = {"xy": 0.7, "xw": 0.4}
        vertices, edges = slice_tesseract(angles_for_vertices)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "slice.json"
            export_to_json(vertices, edges, angles={}, filename=out)
            data = json.loads(out.read_text())

            expected_centroid = np.mean(vertices, axis=0)
            exported_centroid = np.array(data["analysis"]["centroid"])
            self.assertTrue(np.allclose(exported_centroid, expected_centroid))

    def test_export_to_obj_writes_vertices_and_edges(self):
        angles = {"xy": 0.2}
        vertices, edges = slice_tesseract(angles)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "slice.obj"
            export_to_obj(vertices, edges, out)

            self.assertTrue(out.exists())
            lines = out.read_text().splitlines()
            v_lines = [line for line in lines if line.startswith("v ")]
            l_lines = [line for line in lines if line.startswith("l ")]

            self.assertEqual(len(v_lines), len(vertices))
            self.assertGreater(len(l_lines), 0)


if __name__ == "__main__":
    unittest.main()
