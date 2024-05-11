# vector index

## 构建

### c++

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

cmake --build build
```

## 运行

### c++用例

```bash
./build/example/dehnsw_example path/to/dataset/train path/to/dataset/test path/to/dataset/neighbors
```
