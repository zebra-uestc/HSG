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
        non_neighbors_indices = np.random.choice(
            list(non_neighbors_indices), size=k, replace=False
        )
    # 保存为二进制文件
    with open("non_neighbors_indices.bin", "wb") as f:
        f.write(non_neighbors_indices.tobytes())
    print("二进制文件保存成功...\n")
    """
    加载二进制文件，恢复为原来的数组
    with open('non_neighbors_indices.bin', 'rb') as f:
        loaded_indices = np.frombuffer(f.read(), dtype=non_neighbors_indices.dtype)
    print("加载完毕")
    """
    return non_neighbors_indices


def get_mixed_indices(neighbors, train_size, k):
    # 将所有的邻居索引放入一个集合中
    neighbors_set = set(neighbors.flatten())
    # 创建一个包含所有训练集索引的集合
    all_indices_set = set(np.arange(train_size))
    # 找出在训练集中但不在邻居集合中的索引
    non_neighbors_indices = all_indices_set - neighbors_set
    # 随机选择k/2个非邻居索引
    non_neighbors_sample = np.random.choice(
        list(non_neighbors_indices), size=k // 2, replace=False
    )
    # 创建一个空数组用于存储邻居索引
    neighbors_sample = np.empty((0))

    # 从neighbors里选取k/2个索引
    # 存储所有被抽到的元素
    element_list = []
    # 已经抽取的元素总个数
    sum = 0
    # 记录每行剩余的元素个数
    array = np.full(neighbors.shape[0], neighbors.shape[1])
    # 开始抽取元素，直至抽取总数达到k/2
    while sum < k / 2 and np.all(array >= 10):
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
            if array[row] - 1 < 10:
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
    # 合并两个样本以得到混合的索引
    mixed_indices = np.concatenate([non_neighbors_sample, neighbors_sample])
    # 保存为二进制文件
    with open("mixed_indices.bin", "wb") as f:
        f.write(mixed_indices.tobytes())
    print("二进制文件保存成功...\n")
    """
    加载二进制文件，恢复为原来的数组
    with open('mixed_indices.bin', 'rb') as f:
        loaded_indices = np.frombuffer(f.read(), dtype=non_neighbors_indices.dtype)
    print("加载完毕")
    """
    return mixed_indices


def get_topk_indices(neighbors, k):
    # 创建一个空数组用于存储邻居索引
    _topk_indices = np.empty((0))
    # 存储所有被抽到的元素
    element_list = []
    # 已经抽取的元素总个数
    sum = 0
    # 记录每行剩余的元素个数
    array = np.full(neighbors.shape[0], neighbors.shape[1])
    # 开始抽取元素
    for i in range(k):
        top_k_elements = neighbors[:, i : i + 1]
        for j in top_k_elements.flatten():
            # j没有被抽到过，则开始判断是否会导致某行少于10
            if j not in element_list:
                # 在neighbors集中，找到该被抽取到的元素所在行
                rows, _ = np.where(neighbors == j)
                for row in rows:
                    if array[row] - 1 < 10:
                        print("该数据集最多只能选取top", k - 1, "个索引...\n")
                        return
        # 如果所有j都不会导致某行少于10，则将此次选择的结果加入列表
        element_list.extend(top_k_elements.tolist())
        print("已选择", i + 1, "列...\n")
    # 将列表转换为np数组
    topk_indices = np.array(element_list)
    # 保存为二进制文件
    with open("topk_indices.bin", "wb") as f:
        f.write(topk_indices.tobytes())
    print("二进制文件保存成功...\n")
    return topk_indices


k = 100
# # 调用dataset_transform函数，将HDF5文件的内容转换为训练集、测试集和邻居集
f = h5py.File("sift-128-euclidean.hdf5", "r")
train, test, neighbors = dataset_transform(f)
# 找出在neighbors中没有出现过的索引
non_neighbors_indices = get_non_neighbors_indices(neighbors, len(train), k)
# k/2个非邻居索引，k/2个邻居索引
mixed_indices = get_mixed_indices(neighbors, len(train), k)
# 每行前k个
topk_indices = get_topk_indices(neighbors, 10)
