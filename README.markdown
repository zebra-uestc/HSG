# vector index

## 构建

### c++

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

cmake --build build
```

### python

```bash
cd python_binding

g++ -std=c++20 -march=native -shared -DNDEBUG -O2 -flto -fpic $(python3 -m pybind11 --includes) dehnswpy.cpp -o dehnswpy$(python3-config --extension-suffix)
```

## 运行

### c++用例

```bash
./build/example/dehnsw_example path/to/dataset/train path/to/dataset/test path/to/dataset/neighbors
```

### python用例

```bash
cd example

python3 example.py
```
