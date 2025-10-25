
![](./resources/logo/geometric-red.png)

# Install guide

```
git clone --recursive
pip install -r requirements.txt
conda install -c conda-forge libstdcxx-ng
```

build reggrow extension:
```
cd ./segmentation/reggrow/
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

build cut-pursuit extension (instructions from [gitlab repo](https://gitlab.com/1a7r0ch3/parallel-cut-pursuit))
```
cd ./segmentation/cutpursuit/python
python setup.py build_ext
```
