
<p align="center">
  <img src="resources/logo/geometric-red.png" alt="geon logo" width="350">
</p>


## Introduction
*geon* is a research tool for managing and annotating large point cloud datasets stored in HDF5 file containers. 


## Install guide
The tool is tested with Python 3.10 to 3.12 and under *Windows*, *Linux (Ubuntu)* and *MacOS*.

It requires `vtk` and `PyQt6`.


To install run:
```
pip install geon
```

## Development build
Native extensions are built with CMake via scikit-build-core. You will need a C++17 compiler and CMake. Make sure to clone with `--recursive` to pull submodules.

```
git clone --recursive https://github.com/biophase/geon.git
pip install -e .
```

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
- OpenMP is used for some kernels when available.

## Running Geon
```
python -m geon.app
```
