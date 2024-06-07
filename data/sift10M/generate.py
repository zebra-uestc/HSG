import random
import struct
from typing import List, Tuple, Union
import h5py
import numpy as np
import sys


def ivecs(path):
    with open(path, "rb") as file:
        k = int.from_bytes(file.read(4), byteorder="little")
        file.seek(0, 2)
        end = file.tell()
        number = np.uint64(end / ((k + 1) * 4))
        a = np.empty([number, k], dtype=int)
        file.seek(0, 0)
        for i in range(0, number):
            a[i] = np.fromfile(path, count=k, offset=4, dtype=np.int32)
    file.close()
    return a


def bvecs(path, n=0):
    with open(path, "rb") as file:
        k = int.from_bytes(file.read(4), byteorder="little")
        file.seek(0, 2)
        end = file.tell()
        number = np.uint64(end / (k + 4))
        if n == 0 or number < n:
            n = number
        a = np.empty([n, k], dtype=np.uint8)
        file.seek(0, 0)
        for i in range(0, n):
            a[i] = np.fromfile(path, count=k, offset=4, dtype=np.uint8)
    file.close()
    return a


def get_irrelevant(neighbors, train_size, percentage):
    # 将所有的邻居索引放入一个集合中
    neighbors_set = set(neighbors.flatten())
    # print(neighbors_set)
    # 创建一个包含所有训练集索引的集合
    all_indices_set = set(np.arange(train_size))
    # 找出在训练集中但不在邻居集合中的索引
    non_neighbors_indices = all_indices_set - neighbors_set
    k = (int)(len(non_neighbors_indices) * percentage)
    # 如果非邻居索引的数量大于k，那么随机选择k个
    if len(non_neighbors_indices) > k:
        non_neighbors_indices = np.random.choice(
            list(non_neighbors_indices), size=k, replace=False
        )
    # 保存为二进制文件
    while percentage != (float)(int(percentage)):
        percentage *= 10
    with open("delete{0}irrelevant".format(int(percentage)), "wb") as file:
        file.write(struct.pack("Q", k))
        for i in non_neighbors_indices:
            file.write(struct.pack("Q", i))
    file.close()


def get_relevant(neighbors, lower_limit):
    nl = neighbors.tolist()
    # 记录每行选中的个数
    array = np.full(len(neighbors), 0, dtype="uint64")
    while np.any(array < lower_limit):
        row = np.argmin(array)
        # 从所有元素中随机抽取一个
        selected_element = random.choice(nl[row])
        # 在neighbors集中，找到该被抽取到的元素所在行
        rows, _ = np.where(np.asarray(nl) == selected_element)
        for row in rows:
            array[row] += 1
            nl[row].remove(selected_element)
    # 将所有的邻居索引放入一个集合中
    result = set()
    for i in nl:
        for j in i:
            result.add(j)
    # 保存为二进制文件
    with open("delete{0}relevant".format(lower_limit), "wb") as file:
        file.write(struct.pack("Q", len(result)))
        for i in result:
            file.write(struct.pack("Q", i))
    file.close()


# train = bvecs("bigann_base.bvecs", 10000000)
neighbors = ivecs("gnd/idx_10M.ivecs")
get_irrelevant(neighbors, 10000000, 0.75)
get_relevant(neighbors, 100)
