# 4D Rotator

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)

> Note: This project is early work and has been superseded by [4D-Tesseractinator](https://github.com/mbagalman/4D-Tesseractinator), which is the active project and the better version going forward. This repository will remain public in case the earlier work is useful to anyone.

Explore slices of a rotating 4D hypercube (tesseract) in 3D.

## Features

- Build and rotate a 4D tesseract in any of the 6 rotation planes (`xy`, `xz`, `yz`, `xw`, `yw`, `zw`)
- Slice along a fixed `w` hyperplane
- Visualize slices with matplotlib
- Export geometry to JSON and OBJ
- Analyze slice metrics (centroid, bounding box, edge statistics)
- Run gallery and interactive Jupyter demos

## Installation

```bash
git clone https://github.com/mbagalman/Four_D_Rotator.git
cd Four_D_Rotator
```

Core install (math + export + analysis):

```bash
pip install .
```

Install with visualization extras:

```bash
pip install ".[viz]"
```

Install full convenience environment from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Quick Usage

```python
from Four_D_Rotator.geometry import slice_tesseract
from Four_D_Rotator.io_json import export_to_json

angles = {"xy": 0.6, "xz": 1.1, "xw": 0.9}
vertices, edges = slice_tesseract(angles)
export_to_json(vertices, edges, angles, filename="my_hypercube_slice.json")
```

Interactive demo (requires viz dependencies):

```python
from Four_D_Rotator.demos import interactive_demo
interactive_demo()
```

Binder notebook:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mbagalman/Four_D_Rotator/main?labpath=Four_D_Rotator_Demo.ipynb)

## Testing

Run the local test suite:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

GitHub Actions also runs tests on push and pull request.

## License

MIT. See `LICENSE.md`.
