# SIFT

## 数据集规模

| 维度 | 训练集 | 测试集 | 邻居数量 |
|-----|-----------|--------|-----|
| 128 | 1,000,000 | 10,000 | 100 |

## 下载数据集

```bash
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5

# or

wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
```

## 导出数据

仅使用该项目自带的c++案例时需要

```bash
python3 sift.py
```
