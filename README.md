
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

The file format is now version 5 (mesh fields and object membership). Older absolute-stop maps are read with explicit
limits that preserve their previous appearance. Map-name conflicts are resolved
with numbered suffixes and reported when the dataset is populated.


## Mesh reconstruction and IFC

Choose **Document → Mesh → From PCD instance field** with a point cloud active.
Choose the primitive instance field and any additional fields to transfer. Two
additional field selectors are provided initially; add or remove selectors as
needed. The grouped reconstruction controls start with
[`resources/reconstruction_settings.json`](resources/reconstruction_settings.json).
Reconstruction runs in a background worker. Cancel discards the preview; Accept
adds one mesh with normal undo/redo.

The preview ribbon offers **Corners**, face-field display, mesh transparency and
source-cloud visibility. Borders are teal for shared analytic intersections,
green for coplanar joins and red for optimized outer boundaries. **Deviation
(temporary)** shows local RMS point-to-surface distance on a regular surface
sampling, capped at 80,000 vertices. Its Color Channels histogram counts the
represented **source points**, uses numeric distance ticks in metres, and supports
temporary color-map edits. Sampling, distance arrays and border actors are removed
when the preview ends; they do not increase the saved mesh's resolution.
Meshes larger than the sampling budget retain their ordinary field preview.

Each mesh supports vertex, edge, face and object fields using the existing
`FieldType`, semantic schemas and color maps. Use the mesh ribbon to switch levels,
edit fields, select elements or classify them. Click selects an element; Shift-click
toggles it. Object membership is an explicit face-to-object ID array. Edges are
unique undirected vertex pairs. Reconstructed object membership initially follows
the primitive IDs. Fields and schemas persist in HDF5; older geometry-only mesh
files remain readable.

Field transfer happens after simplification and stitching. Source points vote
within their primitive's final faces; semantic and instance fields use majority
voting (ties choose the lowest ID). Numeric, vector and color fields use averages.
Faces without source votes inherit the nearest point from their own primitive.
Deviation is measured to the assigned final triangle, including simplified border
changes; only source points belonging to retained reconstructed primitives count.

Choose **Document → Mesh → Export IFC 4.3…** after accepting the mesh. Select a
semantic field and an instance field separating objects. Object-level semantic
fields can use the mesh's built-in object membership instead. Labels at other
levels vote using the distinct original elements belonging to each object, rather
than duplicating shared vertices. The exporter interprets coordinates as metres.

The class table lists concrete IFC4.3 element and spatial classes, plus their
predefined types. The bundled [`resources/ifc_map.json`](resources/ifc_map.json)
contains defaults for all 21 classes in `BRIDGE_`; load or save an edited mapping
from the export dialog. Mappings include field levels, so identically named
fields at different levels remain distinct. Surface features attach to the
nearest exported pavement/slab or physical host; if none exists, an unspecified
proxy host is created. The IFC model is validated before replacing the target file.

An optional instance field preserves face groups as named
[IFC shape aspects](https://standards.buildingsmart.org/IFC/DEV/IFC4_3/HTML/lexical/IfcShapeAspect.html).
It does not control edge visibility in IFC viewers. Open reconstructions are
exported as [polygonal face sets](https://docs.ifcopenshell.org/autoapi/ifcopenshell/api/geometry/add_mesh_representation/index.html),
without asserting that they are watertight solids. SciPy, Shapely, Matplotlib and
IfcOpenShell are installed with geon's package dependencies.
