import random
from typing import List, Tuple, Union
import h5py
import numpy as np
import struct


def dataset_transform(
    dataset: h5py.Dataset,
) -> Tuple[
    Union[np.ndarray, List[np.ndarray]],
    Union[np.ndarray, List[np.ndarray]],
    Union[np.ndarray, List[np.ndarray]],
]:
    return (
        np.array(dataset["train"]),
        np.array(dataset["test"]),
        np.array(dataset["neighbors"]),
    )


def get_irrelevant(neighbors, train_size, percentage):
    # 将所有的邻居索引放入一个集合中
    neighbors_set = set(neighbors.flatten())
    # 创建一个包含所有训练集索引的集合
    all_indices_set = set(np.arange(train_size))
    # 找出在训练集中但不在邻居集合中的索引
    non_neighbors_indices = all_indices_set - neighbors_set
    print("irrelevant vectors number: {0}".format(len(non_neighbors_indices)))
    k = (int)(len(non_neighbors_indices) * percentage)
    print("deleted irrelevant vectors number: {0}".format(k))
    non_neighbors_indices = np.random.choice(
        list(non_neighbors_indices), size=k, replace=False
    )
    # 保存为二进制文件
    while percentage != (float)(int(percentage)):
        percentage *= 10
    with open("delete{0}irrelevant.binary".format(int(percentage)), "wb") as file:
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
        for row in range(0, len(neighbors)):
            if selected_element in nl[row]:
                array[row] += 1
                nl[row].remove(selected_element)
    # 将所有的邻居索引放入一个集合中
    result = set()
    for i in nl:
        for j in i:
            result.add(j)
    # 保存为二进制文件
    with open("save{0}relevant.binary".format(lower_limit), "wb") as file:
        file.write(struct.pack("Q", len(result)))
        print("deleted vectors number: {0}".format(len(result)))
        for i in result:
            file.write(struct.pack("Q", i))
    file.close()


train, _, neighbors = dataset_transform(
    h5py.File("fashion-mnist-784-euclidean.hdf5", "r")
)
get_irrelevant(neighbors, len(train), 0.75)
get_relevant(neighbors, 10)
