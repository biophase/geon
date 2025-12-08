
![](./resources/logo/geometric-red.png)

# Install guide
## Libraries
```
git clone --recursive https://github.com/biophase/geon
cd geon
conda create -n geon python=3.10
conda activate geon
pip install -r requirements.txt
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

install pytorch geometric and its dependencies (Linux, for Windows see below)
```
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/pyg_lib-0.4.0%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_cluster-1.6.3%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_scatter-2.1.2%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_sparse-0.6.18%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_spline_conv-1.2.2%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
```

install pytorch geometric for Windows
```
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/pyg_lib-0.4.0%2Bpt25cu121-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_cluster-1.6.3%2Bpt25cu121-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_scatter-2.1.2%2Bpt25cu121-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_sparse-0.6.18%2Bpt25cu121-cp310-cp310-win_amd64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_spline_conv-1.2.2%2Bpt25cu121-cp310-cp310-win_amd64.whl
```

## Build C++ extensions
Install cmake, if missing:
on Windows:
```
winget install Kitware.CMake
```

on Ubuntu
```
sudo apt update
sudo apt install cmake
conda install -c conda-forge libstdcxx-ng
```

build reggrow extension:
```
cd ./segmentation/reggrow/
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ../../..
```

build cut-pursuit extension (instructions from [gitlab repo](https://gitlab.com/1a7r0ch3/parallel-cut-pursuit))
```
python segmentation/cutpursuit/python/setup.py build_ext
python segmentation/cutpursuit/pcd-prox-split/grid-graph/python/setup.py build_ext
```

## Install Geon
```
pip install -e .
```

## Running Geon
```
python -m geon.main
```


# HDF File structure:
```
.
└── /document
    ├── attrs:
    │   ├── geon_format_version: <GEON_FORMAT_VERSION>
    │   ├── type:
    │   └── name:
    └── PCD_000 (Group)
        ├── attrs:
        │   ├── type_id
        │   └── id
        ├── points (Dataset, shape(N,3))
        └── fields (Group)
            ├── intensity (Group)
            │   ├── data (Dataset, shape (N,1))
            │   ├── attrs:
            │   │   └── field_type: "INTENSITY"
            │   └── color_map (Dataset, optional)
            └── semantics (Group)
                ├── data (Dataset, shape (N,))
                ├── attrs
                │   └── field_type: "SEMANTIC"
                └── semantic_schema (Dataset, scalar string, JSON)
                    └── attrs:
                        └── name: "schema_name"

```