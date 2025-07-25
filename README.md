# 4D Rotator

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)

Explore the geometry of a rotating hypercube—also known as a tesseract—sliced into 3D space. Perfect for data scientists, educators, and geometry nerds who always suspected the fourth dimension was just shy.

This project offers clean, well-commented Python code to:

- Define a 4D cube (tesseract)
- Rotate it in any of the 6 rotation planes (e.g., `xy`, `xw`, etc.)
- Slice it into 3D space for visualization
- Render the resulting geometry with matplotlib
- Save interesting configurations to JSON
- Run interactive demos in a Jupyter notebook

---

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/mbagalman/Four_D_Rotator.git
cd Four_D_Rotator
```

### 2. Install the package

```bash
pip install .
```

---

## 📦 Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- ipywidgets (for interactive demo)

You can install them all with:

```bash
pip install -r requirements.txt
```

---

## 📓 Try It in Jupyter

Launch the interactive viewer:

```python
from Four_D_Rotator.demos import interactive_demo
interactive_demo()
```

This will display sliders for each rotation plane. Explore how different angles affect the 3D slice. It's like flying a tesseract with knobs.

Or use Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mbagalman/Four_D_Rotator/main?labpath=Four_D_Rotator_Demo.ipynb)

---

## 🎨 Save Your Favorite Slice

Find a rotation you like using the sliders! Then, copy the angle values into the `my_angles` dictionary below and run this code to save your unique creation:

```python
from Four_D_Rotator.io_json import export_to_json
from Four_D_Rotator.geometry import slice_tesseract

my_angles = {
    "xy": 0.6,
    "xz": 1.1,
    "xw": 0.9,
}

vertices, edges = slice_tesseract(my_angles)
export_to_json(vertices, edges, my_angles, filename="my_hypercube_slice.json")
```

---

## 🤖 CLI and Gallery Support (Coming Soon)

Stay tuned for command-line utilities to generate animations and a gallery of iconic tesseract slices. Suggestions welcome!

---

## 🧠 Credits & License

Created by Michael Bagalman. MIT License.

And yes, we considered calling this project *Mostly Harmless Geometry*.


