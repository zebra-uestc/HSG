from typing import List, Tuple, Union
import h5py
import numpy as np
import struct


# 定义函数，将HDF5格式的数据集转换为numpy格式的数据集
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


def get_relevant(neighbors, k, lower_limit):
    # 将所有的邻居索引放入一个集合中
    neighbors_set = set(neighbors.flatten())
    # 创建一个空数组用于存储邻居索引
    neighbors_sample = np.empty((0))
    # 从neighbors里选取k个索引
    # 存储所有被抽到的元素
    element_list = []
    # 已经抽取的元素总个数
    sum = 0
    # 记录每行剩余的元素个数
    array = np.full(neighbors.shape[0], neighbors.shape[1])
    # 开始抽取元素，直至抽取总数达到k
    while sum < k and np.all(array >= lower_limit):
        willApend = True
        # 从所有元素中随机抽取一个
        selected_element = np.random.choice(list(neighbors_set), size=1, replace=False)
        data_choie = int(selected_element.item())
        # 在neighbors集中，找到该被抽取到的元素所在行
        rows, _ = np.where(neighbors == data_choie)
        # 如果被抽取的元素之前被抽到过，则重新选择
        if data_choie in element_list:
            willApend = False
        # 选出的元素使得至少有一行剩余元素少于10个，则重新选择
        for row in rows:
            if array[row] - 1 < lower_limit:
                willApend = False
        # 如果抽取的元素有效，则添加进列表
        if willApend:
            # 添加进抽取元素列表
            element_list.append(data_choie)
            # 列表元素个数加一
            sum += 1
            # 所在行元素个数减一
            for row in rows:
                array[row] -= 1
    # 将列表转换为np数组
    neighbors_sample = np.array(element_list)
    # 保存为二进制文件
    with open("delete{0}relevant".format(k), "wb") as file:
        file.write(struct.pack("Q", k))
        for i in neighbors_sample:
            file.write(struct.pack("Q", i))
    file.close()
    # with open("mixed_indices.bin", "wb") as f:
    #     f.write(mixed_indices.tobytes())
    # print("二进制文件保存成功...\n")
    """
    加载二进制文件，恢复为原来的数组
    with open('mixed_indices.bin', 'rb') as f:
        loaded_indices = np.frombuffer(f.read(), dtype=non_neighbors_indices.dtype)
    print("加载完毕")
    """
    # return mixed_indices


# 调用dataset_transform函数，将HDF5文件的内容转换为训练集、测试集和邻居集
f = h5py.File("gist-960-euclidean.hdf5", "r")
train, test, neighbors = dataset_transform(f)
# 找出在neighbors中没有出现过的索引
get_irrelevant(neighbors, len(train), 0.75)
get_relevant(neighbors, 30, 10)
