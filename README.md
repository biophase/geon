
![](./resources/logo/geometric-red.png)

# Install guide

```
git clone --recursive
conda create -n geon python=3.10
conda activate geon
pip install -r requirements.txt
conda install -c conda-forge libstdcxx-ng
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

install pytorch geometric and its dependencies
```
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/pyg_lib-0.4.0%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_cluster-1.6.3%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_scatter-2.1.2%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_sparse-0.6.18%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_spline_conv-1.2.2%2Bpt25cu121-cp310-cp310-linux_x86_64.whl
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
cd ./segmentation/cutpursuit/python
python setup.py build_ext
```
