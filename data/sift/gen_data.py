from typing import List, Tuple, Union
import h5py
import numpy as np
import struct

# 定义函数，将HDF5格式的数据集转换为numpy格式的数据集
def dataset_transform(dataset: h5py.Dataset) -> Tuple[
    Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]:
    return np.array(dataset["train"]), np.array(dataset["test"]), np.array(dataset["neighbors"])

def get_non_neighbors_indices(neighbors, train_size, k):
    # 将所有的邻居索引放入一个集合中
    neighbors_set = set(neighbors.flatten())
    # print(neighbors_set)
    # 创建一个包含所有训练集索引的集合
    all_indices_set = set(np.arange(train_size))
    # 找出在训练集中但不在邻居集合中的索引
    non_neighbors_indices = all_indices_set - neighbors_set
    # 如果非邻居索引的数量大于k，那么随机选择k个
    if len(non_neighbors_indices) > k:
        non_neighbors_indices = np.random.choice(list(non_neighbors_indices), size=k, replace=False)
    # 保存为二进制文件
    with open('non_neighbors_indices.bin', 'wb') as f:
        f.write(non_neighbors_indices.tobytes())
    print("二进制文件保存成功...\n")
    '''
    加载二进制文件，恢复为原来的数组
    with open('non_neighbors_indices.bin', 'rb') as f:
        loaded_indices = np.frombuffer(f.read(), dtype=non_neighbors_indices.dtype)
    print("加载完毕")
    '''
    return non_neighbors_indices

def get_mixed_indices(neighbors, train_size, k):
    # 将所有的邻居索引放入一个集合中
    neighbors_set = set(neighbors.flatten())
    # 创建一个包含所有训练集索引的集合
    all_indices_set = set(np.arange(train_size))
    # 找出在训练集中但不在邻居集合中的索引
    non_neighbors_indices = all_indices_set - neighbors_set
    # 随机选择k/2个非邻居索引
    non_neighbors_sample = np.random.choice(list(non_neighbors_indices), size=k//2, replace=False)
    # 随机选择k/2个邻居索引
    neighbors_sample = np.random.choice(list(neighbors_set), size=k//2, replace=False)
    # 合并两个样本以得到混合的索引
    mixed_indices = np.concatenate([non_neighbors_sample, neighbors_sample])
    # 保存为二进制文件
    with open('mixed_indices.bin', 'wb') as f:
        f.write(non_neighbors_indices.tobytes())
    print("二进制文件保存成功...\n")
    '''
    加载二进制文件，恢复为原来的数组
    with open('mixed_indices.bin', 'rb') as f:
        loaded_indices = np.frombuffer(f.read(), dtype=non_neighbors_indices.dtype)
    print("加载完毕")
    '''
    return mixed_indices

k = 100
# # 调用dataset_transform函数，将HDF5文件的内容转换为训练集、测试集和邻居集
f = h5py.File('sift-128-euclidean.hdf5', 'r')
train, test, neighbors = dataset_transform(f)
# 找出在neighbors中没有出现过的索引
non_neighbors_indices = get_non_neighbors_indices(neighbors, len(train), k)
mixed_indices = get_mixed_indices(neighbors, len(train), k)
