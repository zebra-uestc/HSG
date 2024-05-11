from typing import List, Tuple, Union
import h5py
import numpy as np
import struct

# 定义函数，将HDF5格式的数据集转换为numpy格式的数据集
def dataset_transform(dataset: h5py.Dataset) -> Tuple[
    Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]:
    """
    Transforms the dataset from the HDF5 format to conventional numpy format.

    If the dataset is dense, it's returned as a numpy array.
    If it's sparse, it's transformed into a list of numpy arrays, each representing a data sample.

    Args:
        dataset (h5py.Dataset): The input dataset in HDF5 format.

    Returns:
        Tuple[Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]: Tuple of training and testing data in conventional format.
    """
    # 打印数据集中的每个元素
    print('数据集中的元素：')
    for i in dataset:
        print(i, end='\n')
    # 打印数据集的属性
    print('数据集的属性：')
    for i in dataset.attrs:
        print(i, end='\n')
    # 打印数据集的'distance'属性
    print('数据集的distance属性：',dataset.attrs.get("distance"))
    print('打印结束'+'\n')
    # 打印训练集的长度、每一个元素的长度和第一个元素的类型
    print('训练集的长度：',len(np.array(dataset["train"])))
    print('每个元素的长度：',len(np.array(dataset["train"])[0]))
    print('第一个元素的类型',type(np.array(dataset["train"])[0][0]))
    # 打印测试集的长度、每一个元素的长度和第一个元素的类型
    print('测试集的长度：',len(np.array(dataset["test"])))
    print('每个元素的长度：',len(np.array(dataset["test"])[0]))
    print('第一个元素的类型',type(np.array(dataset["test"])[0][0]))
    # print('打印结束'+'\n')
    # 打印邻居集的长度、第一个元素的长度和第一个元素的类型
    print('邻居集的长度：',len(np.array(dataset["neighbors"])))
    print('每个元素的长度：',len(np.array(dataset["neighbors"])[0]))
    print('第一个元素的类型',type(np.array(dataset["neighbors"])[0][0]))
    # 返回转换后的训练集合、测试集和邻居集
    return np.array(dataset["train"]), np.array(dataset["test"]), np.array(dataset["neighbors"])

# 调用dataset_transform函数，将HDF5文件的内容转换为训练集、测试集和邻居集
f = h5py.File('sift-128-euclidean.hdf5', 'r')
train, test, neighbors = dataset_transform(f)
# 创建并打开一个名为train的二进制文件，将训练集写入其中
with open('train', 'wb') as file:
    # 写入训练集长度
    file.write(struct.pack("Q", len(train)))
    print('训练集长度：',len(train))
    # 写入训练集第一个元素的大小
    file.write(struct.pack("Q", train[0].size))
    print('训练集第一个元素的大小：',train[0].size)
    # 遍历训练集的每个元素，并将其写入文件
    for i in train:
        for j in i:
            file.write(struct.pack("f", j))
file.close()
with open('test', 'wb') as file:
    file.write(struct.pack("Q", len(test)))
    print(len(test))
    file.write(struct.pack("Q", test[0].size))
    print(test[0].size)
    for i in test:
        for j in i:
            file.write(struct.pack("f", j))
file.close()
with open('neighbors', 'wb') as file:
    file.write(struct.pack("Q", len(neighbors)))
    print(len(neighbors))
    file.write(struct.pack("Q", neighbors[0].size))
    print(neighbors[0].size)
    for i in neighbors:
        for j in i:
            file.write(struct.pack("Q", j))
file.close()
