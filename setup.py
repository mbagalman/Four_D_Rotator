from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="Four-D-Rotator",
    version="1.0.0",
    description="Toolkit for rotating and slicing a 4D tesseract into 3D geometry.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Michael Bagalman",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(include=["Four_D_Rotator", "Four_D_Rotator.*"]),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.4",
            "ipywidgets>=7.6",
        ]
    },
)
