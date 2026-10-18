
<p align="center">
  <img src="resources/logo/geometric-red.png" alt="geon logo" width="350">
</p>

<p align="center">
  <a href="https://github.com/biophase/geon/actions/workflows/ci.yml">
    <img src="https://github.com/biophase/geon/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI">
  </a>
  <a href="https://pypi.org/project/geon/">
    <img src="https://img.shields.io/pypi/v/geon" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/geon/">
    <img src="https://img.shields.io/pypi/pyversions/geon" alt="Python versions">
  </a>
  <a href="https://pypi.org/project/geon/">
    <img src="https://img.shields.io/pypi/l/geon" alt="License">
  </a>
</p>


## Introduction
*geon* is a research tool for managing and annotating large point cloud datasets stored in HDF5 file containers. 


## Install guide
The tool is tested with Python 3.10 to 3.12 and under *Windows*, *Linux (Ubuntu)* and *MacOS*.

It requires `vtk` and `PyQt6`.


To isntall from PyPI run:

```
pip install geon
```

Then run the GUI with

```
python -m geon.app
```

### Experimental native Wayland rendering

On a Wayland desktop, run with the Qt-owned OpenGL surface:

```bash
GEON_QT_RENDERER=opengl QT_QPA_PLATFORM=wayland python -m geon.app
```



## Development build
For development clone and install with
```
git clone --recursive https://github.com/biophase/geon.git
cd geon
pip install -e .
```

Native extensions are built with CMake via scikit-build-core. You will need a C++17 compiler and CMake. 

Building native modules:
- Linux/MacOS:
  ```
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/src/geon/_native"
  cmake --build build -j
  ```
- On Windows (MSVC), use:
  ```
  cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE="%CD%\\src\\geon\\_native"
  cmake --build build --config Release
  ```
- OpenMP is used for some methods when available.



# Color channels

Open **View → Color channels** to inspect visible-point histograms and edit a
field's colors. The dock can float or be hidden; hidden docks defer histogram
calculation. This feature uses the `pyqtgraph` dependency.

Scalar maps use a color gradient; vector maps use three component-to-intensity
curves. Normals default to **RGB (-1 to 1)**. Color fields default to **RGB (0 to
1)** or **RGB (0 to 255)** according to their values. Empty range limits use the
full field/component extrema, independently of point visibility.

Drag interior stops, double-click the gradient to add a stop, or double-click a
stop to edit its output. End stops cannot be removed. **Duplicate** creates an
independent map. Edits preview on the active field; **Revert** restores its
original settings. **Save** updates matching named map definitions throughout
the dataset, writing only color-map metadata to existing HDF5 files. Component
bindings belong to each field. Unsaved documents still require a document save.

The file format is now version 4. Older absolute-stop maps are read with explicit
limits that preserve their previous appearance. Map-name conflicts are resolved
with numbered suffixes and reported when the dataset is populated.

