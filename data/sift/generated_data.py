from typing import List, Tuple, Union
import h5py
import numpy as np
import struct

# 定义函数，将HDF5格式的数据集转换为numpy格式的数据集
def dataset_transform(dataset: h5py.Dataset) -> Tuple[
    Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]:
    return np.array(dataset["train"]), np.array(dataset["test"]), np.array(dataset["neighbors"])


def save_diff_vectors(dataset_file, output_file, id_output_file, k):
    # 打开HDF5文件
    f = h5py.File(dataset_file, 'r')
    # 调用dataset_transform函数，将HDF5文件的内容转换为train集、test集和neighbors集
    train, test, neighbors = dataset_transform(f)
    print('数据集分类完成...\n')
    # 将训练集和邻居集转换为字典，以便进行快速查找，键是元组形式的数据，值是数据的ID
    train_dict = {tuple(i): idx for idx, i in enumerate(train)}
    neighbors_set = set(tuple(i) for i in neighbors)
    # 找出在训练集中但不在邻居集中的向量
    diff_dict = {i: train_dict[i] for i in train_dict.keys() - neighbors_set}
    # 将结果转换回numpy数组
    diff_array = np.array([list(i) for i in diff_dict.keys()])
    diff_ids = list(diff_dict.values())
    print('开始提取...\n')
    # 将结果保存为二进制文件
    with open(output_file, 'wb') as file:
        count = 0
        file.write(struct.pack("Q", len(diff_array)))
        file.write(struct.pack("Q", diff_array[0].size))
        for i in diff_array:
            if count >= k:
                break
            for j in i:
                file.write(struct.pack("f", j))
            count += 1
            print('生成',count,'条数据...\n')
            print('第',count,'条数据为：',i,'\n')
    with open(id_output_file, 'wb') as file:
        for id in diff_ids[:k]:
            file.write(struct.pack("Q", id))
# 使用函数
save_diff_vectors('sift-128-euclidean.hdf5', 'diff', 'diff_ids', 100)