
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


## Experimental planar region growing on superpoints

Choose **Document → Point cloud → Segmentation → Planar region growing on
superpoints (experimental)**. Select an existing instance field (for example,
the output of Superpoint segmentation). Each nonnegative instance ID is one
indivisible superpoint; negative labels are excluded. **On selection only**
expands selected points to their complete source superpoints. Results go into a
new INSTANCE field, defaulting to `superpoint_planar_regions`; excluded or
unassigned points receive `-1`. Cancel discards the run. Save the document to
persist the new field.

Two superpoints are neighbors if any member-point pair is at distance
`<= min_d` (default `0.05`). Candidate acceptance tests its centroid's distance
to the current region plane against `epsilon × epsilon_multiplier` (defaults
`0.03 × 3`) and compares its PCA normal with the region normal (default angle
`29°`). The angular tolerance tightens with distance from the region centroid,
reaching zero at `max_dist_from_cent` (default `50`). All distances use the
cloud's coordinate units. Normals are unoriented; no point-normal field is used.

Seeds are tried deterministically, most planar first. Seed gating defaults to
at least 10 member points, planarity `(λ₂−λ₁)/λ₃ >= 0.20`, and scattering
`λ₁/λ₃ <= 0.35`, with ascending PCA eigenvalues. Gating applies only to seeds;
every candidate must have at least three noncollinear points for its PCA normal.
Degenerate superpoints remain unassigned. Regions are refitted from all member
points using cached statistics, so large superpoints contribute proportionally.
`tau` (default `80`) counts original points; `first_refit` (default `4`) counts
superpoints, and subsequent thresholds grow by `refit_multiplier` (default `2`).
Pruning removes whole superpoints and splits disconnected graph components.
Each source ID remains indivisible even if its own points are disconnected.

The Python prototype is also importable without Qt or the native extension:

```python
from geon.algorithms.superpoint_region_growing import segment_planar_superpoints

labels, stats = segment_planar_superpoints(
    cloud, instance_field_name="superpoints",
    params={"min_d": 0.05, "epsilon": 0.03, "tau": 80},
)
# Alternatively: coordinates plus instance_labels shaped (N,) or (N, 1).
# selection_indices optionally expands to complete superpoints.
```

The function leaves its inputs unchanged and returns full-length `int32` labels
and diagnostics (superpoints, edges, invalid normals, seed attempts, regions,
unassigned points, and runtime). Optional `progress_callback(stage, done, total)`
and `cancel_requested()` support background callers; cancellation raises
`SuperpointGrowthCancelled`. Parameters persist separately under
`superpoint_region_growing__` in the preferences TOML. There is no chunk
reconciliation, reassignment, or native probabilistic stopping in this version;
results need not match point-based region growing.

Touching graphs use the native streaming backend when installed. A C++ radius
search inserts canonical superpoint pairs directly into a hash set, without
materializing point-pair arrays. It releases the GIL and checks cancellation
between query batches. Final edge keys are sorted for deterministic output.
Homogeneous spatial-tree branches are skipped for the query's own label or
an already confirmed superpoint edge, avoiding repeated dense interior work.
Local Windows Release benchmarks on 10 million synthetic points measured
13.8 s / 1,012 MiB peak process RSS for the sparse case and 72.0 s / 904 MiB
for the dense case. These include the Python runtime, inputs and graph and
are workload-specific, not expected timings for arbitrary scans.
The spatial index and unique edges still require memory; dense searches still
visit redundant contacts. Zero radius uses exact-coordinate grouping.

Without the native module, touching graphs select an exact fallback. Modest contact counts
use SciPy `query_pairs(eps=0)`, vectorized label mapping, and collision-free
64-bit edge keys deduplicated with NumPy. This path is limited to 32 pairs per
point and a 256 MiB work budget, conservatively reserving 128 bytes per pair.
A partial count can prove that the budget is exceeded before a full count is
needed. Dense contacts use radius-bucketed bounding-box centers, conservative
box rejection, and batched point-to-superpoint KD-tree queries that stop at the
first verified contact. The local tree cache is bounded to 128 MiB. Zero radius
uses exact-coordinate grouping. Neither path downsamples points or approximates
the touching threshold. Graph work still requires Python loops over candidate
superpoint pairs and final edges, rather than every point contact.

Diagnostics include `graph_backend`, `graph_seconds`, `graph_point_pairs`,
`graph_point_pairs_exact`, and `graph_candidate_pairs`. A non-exact pair count is
a lower bound used to select the direct backend (zero/unmeasured for native). The memory budget is for
temporary contact buffers; coordinates, spatial/grouping indices, and the final
graph require additional memory. Cancellation is checked between bounded batches;
individual compiled KD-tree and sorting calls cannot be interrupted. An inherently
dense final graph can still be expensive. Reproduce time and peak-process-memory
measurements with `scripts/benchmark_touching_graph.py` (requires `psutil`), using
`--points 1000000` or `--points 10000000` and `--density sparse` or `--density dense`.

Example Windows/Python 3.12 runs (single runs, with other build work active;
not controlled speedup comparisons or predictions for other scans):

| Points | Contacts | Superpoints | Edges | Backend | Graph time | Peak process RSS |
|---:|---|---:|---:|---|---:|---:|
| 1,000,000 | Sparse | 40,000 | 79,600 | Vectorized | 7.74 s | 168 MiB |
| 1,000,000 | Dense | 2,500 | 9,702 | Direct | 6.18 s | 161 MiB |
| 10,000,000 | Sparse | 400,000 | 798,735 | Direct | 316.79 s | 779 MiB |
| 10,000,000 | Dense | 25,000 | 99,052 | Direct | 42.66 s | 608 MiB |

RSS includes the inputs, indices, graph, and Python runtime. Sparse and dense
fixtures have different superpoint sizes; fewer redundant point contacts does
not imply fewer candidate superpoint pairs or a faster direct graph.

## Mesh reconstruction and IFC

Choose **Document → Mesh → From PCD instance field** with a point cloud active.
Choose the primitive instance field and any additional fields to transfer. Two
additional field selectors are provided initially; add or remove selectors as
needed. The grouped reconstruction controls start with
[`resources/reconstruction_settings.json`](resources/reconstruction_settings.json).
Reconstruction runs in a background worker. Cancel discards the preview; Accept
adds one mesh with normal undo/redo. The **Log** button in the reconstruction
toolbar opens the progress and fit/rejection diagnostics during processing and
preview. Failed runs also expose the log through **Show Details** in the error
dialog; a run with no faces reports the rejected patch IDs and reasons.

The preview ribbon offers **Corners**, face-field display, mesh transparency and
source-cloud visibility. Borders are teal for shared surface intersections,
green for coplanar joins and red for optimized outer boundaries. **Deviation
(temporary)** shows local RMS point-to-surface distance on a regular surface
sampling at the requested spacing, without a default vertex/face cap. Its Color Channels histogram counts the
represented **source points**, uses numeric distance ticks in metres, and supports
temporary color-map edits. Sampling, distance arrays and border actors are removed
when the preview ends; they do not increase the saved mesh's resolution.
Preview generation reports progress and supports cancellation between faces.
Automatic numeric histograms choose bins after filtering to the 1st–99th
percentile range; categorical counts and explicitly selected ranges stay intact.

Coplanar border validation applies each proposal to the current boundary and
checks previously accepted seams, rather than replaying all earlier edits.
Progress is reported within validation. Enabled border regularization tries
the simplified common border first, falling back to the observed border only
if the regularized edit fails validation. Seam preservation permits collinear
subdivision; new seam nodes are synchronized before meshing.

Dense B-spline training uses at most 16,384 deterministic spatially stratified
rows with multiplicity weights. Geometric holdout tests and fit thresholds are
unchanged, and final error validation still includes all source points in chunks.

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

### B-spline fallback and surface intersections

Reconstruction preserves the existing plane/cylinder model selection. When the
plane exceeds `max_plane_rmse` and no cylinder meets its acceptance criteria,
**B-spline fallback** can fit a non-rational piecewise bicubic height patch. It
starts with 4 by 4 control heights and adds spans only if geometric validation
requires them, up to 12 poles along the longer chart axis and 8 along the other
(4 across narrow strips). The reconstruction settings resource retains its plane limit of **0.5**;
lower that limit to make smaller departures from planarity eligible. A suitable
plane or accepted cylinder is never replaced by a spline.

The height chart uses the primitive's PCA axes, so vertical and rotated surfaces
work. If the least-variance direction is unsuitable, the other PCA charts are
tested with the same error and slope requirements. This matters for long curved
girder sides: the transverse bend can vary much more than the web height, making
the ordinary PCA plane nearly horizontal. A single cubic span can also underfit
a long arc even when its curvature is low. The surface must remain single-valued
over the chosen local plane. Robust,
density-balanced least squares, a normalized bending penalty, and deterministic
spatial holdout validation reject poorly supported, noisy, multilayer, steep, or
rank-deficient fits. Validation uses OCCT closest-surface distances; UV chart
projection is a separate operation. Accepted models are refitted using all fitting
samples. These controls appear in the reconstruction dialog:

| Parameter | Default | Meaning |
|---|---:|---|
| `fit_bsplines` | `True` | Enable fallback after plane/cylinder selection |
| `max_bspline_rmse` | `0.025` | Maximum geometric holdout RMSE, in coordinate units |
| `bspline_error_ratio` | `0.5` | Maximum spline/plane holdout RMSE ratio |
| `bspline_smoothing` | `0.001` | Dimensionless normalized bending weight |
| `bspline_max_slope` | `1` | Maximum height slope relative to the local PCA plane |
| `bspline_mesh_spacing` | `0.5` | Maximum regular UV cell width, in coordinate units |

Cylinders also use a regular grid in physical arc length and axial height.
`cylinder_mesh_spacing` defaults to `0.25` coordinate units; the
`cylinder_chord_tolerance` can require smaller cells. Complete cylinders use an
integer number of angular cells so the periodic seam closes. Shared curves and
bridge borders receive matching grid nodes before triangulation.

Near-tangent plane/cylinder boundaries can join through the supported border
pass even without a transverse OCCT intersection. Local normals, border distance
and opposing face interiors must pass the coplanar-join tests. Each narrow bridge
is checked in both faces' own UV charts; an entire curved face is not projected
into another chart, where wrapping or folding could falsely reject the join.

Observed UV footprints and holes remain authoritative. `edge_tolerance` bounds
border extension; `support_step` and `curve_tolerance` control intersection
sampling. Both are in coordinate units. OCCT computes every plane/cylinder/spline
pair and curve-surface three-patch corners. Support and border checks follow
intersection, with shared 3D seams and UV curves retained on both incident patches.
Spline meshing clips a regular UV grid to the footprint and holes, reducing cell
width when needed for geometric chord error. Both incident patches share the
union of seam/grid crossings. Boundary cells can have irregular triangles, but
interior cells no longer form long boundary-to-boundary triangle fans.
The fit preview shows solid purple B-spline UV lines. **Control points / net**
optionally shows orange poles connected by dashed lines. Both overlays can be
toggled independently and are removed when accepting or cancelling the preview;
they are not exported as mesh geometry.
The reconstruction log prefixes each entry with elapsed seconds and the time
since the previous entry (`delta`), measured in the worker before GUI delivery.
The delta belongs to the work between the two messages, not necessarily to the
stage named by the second message. Final diagnostics are labelled separately;
failure messages include the actual exception in the log and error details.
UV grid nodes subdivide already accepted boundary segments without rerunning
border replacement. Alpha-footprint classification uses retained triangles next
to exposed edges, avoiding whole-triangulation point-location preprocessing.
Seams split at deep observed boundary notches; endpoint trimming remains
bounded by `edge_tolerance`, retaining scan gaps instead of dropping the complete
intersection or closing unsupported holes.

Coplanar border validation edits the current boundary incrementally and checks
that earlier shared seams remain covered, including collinear subdivisions at
junctions. Both incident faces receive the same seam subdivision nodes before
meshing. Progress reports the validated and accepted proposal counts. With
`regularize_coplanar_borders` enabled, the common simplified border projected
onto each plane is attempted before the observed-border fallback.

Diagnostics record selected frames, control-grid sizes, errors, rejected fits, failed
intersections, and tangent/coincident ambiguities. The output remains a triangle
mesh and the existing IFC mesh representation; no editable CAD B-rep is stored.

The native `geon._native.surface_geometry` module statically links the geometry
toolkit dependency closure of official **OCCT V7_9_3**, pinned at
`a016080bf6738d6aeae020badee4e888ad1540a5` in `external/opencascade`. Initialize
submodules before building (`git submodule update --init --recursive`), then
rebuild/install geon normally. CMake builds OCCT as a separate ExternalProject,
with position-independent code and isolated symbols; visualization, exchange,
GUI, and optional integrations are disabled. PythonOCC is not required. Licenses
and the OCCT exception ship under `geon/resources/licenses/opencascade`, and the
source distribution includes the pinned source. Wheel tests exercise geometry
alongside IfcOpenShell in both import orders; release builds also run a frozen
PyInstaller geometry smoke test on Windows, Linux, and macOS.

### Merging layers

Highlight two or more whole layer rows of the same type in the Scene browser,
then choose **Layer → Merge** or **Merge** from their right-click menu. The only
dialog asks for the output name and defaults to the first highlighted layer in
browser order. The result replaces the inputs at their first position; Undo
restores the original layers. Geometric point/face selections do not limit a merge.

Point clouds concatenate coordinates and fields; meshes concatenate disjoint
topology and vertex, edge, face, and object fields. Bounding-box lists and cell
complex graphs are copied and concatenated with unique internal IDs. Camera
snapshots cannot be merged. The result inherits the first layer's display settings.
Matching fields are combined by name, type, and component count. Missing labels
are filled with `-1`, missing colours with zero, and other missing numeric values
with `NaN`. Incompatible same-name fields receive unique suffixes. Nonnegative
instance labels are renumbered into disjoint ranges per input, keeping instances
from different layers separate; negative labels stay unassigned. Semantic classes
are aligned by class name, retaining the first
colour. Merging does not weld geometry or run segmentation.

New data and layer subclasses implement `merge(items, *, name=None)` to return
an independent object without modifying inputs. Unsupported data types explicitly
set `supports_merge=False`; layer eligibility is exposed through `can_merge`.
