import time
from typing import List, Tuple, Union
import h5py
import numpy

import sys

sys.path.append("python_binding")  # noqa
import dehnswpy


def calculate_distance(vector1: numpy.ndarray, vector2: numpy.ndarray) -> float:
    distance = 0.0
    for i in range(len(vector1)):
        t = (vector1[i] - vector2[i])
        distance += t * t
    return distance


def get_refernece_answer(train: numpy.ndarray, test: numpy.ndarray, neighbors: numpy.ndarray) -> numpy.ndarray:
    reference: numpy.empty((len(test), len(neighbors)), dtype='float32')
    for i in range(len(test)):
        for j in range(len(neighbors)):
            reference[i][j] = calculate_distance(test[i], train[neighbors[i][j]])
    return reference


def verify(reference_answer: numpy.ndarray, test: int, results: numpy.ndarray) -> int:
    hit = 0
    for distance in reference_answer[test]:
        if (results[hit] <= distance):
            hit += 1
    return hit


def dataset_transform(dataset: h5py.Dataset) -> Tuple[
    Union[numpy.ndarray, List[numpy.ndarray]], Union[numpy.ndarray, List[numpy.ndarray]], Union[
        numpy.ndarray, List[numpy.ndarray]]]:
    for i in dataset:
        print(i, end='\n')
    print()
    for i in dataset.attrs:
        print(i, end='\n')
    print()
    print(dataset.attrs.get("distance"))
    print()
    print(len(numpy.array(dataset["train"])))
    print()
    print(len(numpy.array(dataset["train"])[0]))
    print()
    print(type(numpy.array(dataset["train"])[0][0]))
    print()
    print(len(numpy.array(dataset["test"])))
    print()
    print(len(numpy.array(dataset["test"])[0]))
    print()
    print(type(numpy.array(dataset["test"])[0][0]))
    print()
    print(len(numpy.array(dataset["neighbors"])))
    print()
    print(len(numpy.array(dataset["neighbors"])[0]))
    print()
    print(type(numpy.array(dataset["neighbors"])[0][0]))
    print()

    return numpy.array(dataset["train"]), numpy.array(dataset["test"]), numpy.array(dataset["neighbors"])


rf = open('faiss_hnsw.txt', 'w', encoding='utf-8')
# 数据集路径
# 使用时需要修改
f = h5py.File('/root/host/data/fashion-mnist/fashion-mnist-784-euclidean.hdf5', 'r')
train, test, neighbors = dataset_transform(f)
reference_answer = get_refernece_answer(train, test, neighbors)
n, d = len(train), len(train[0])
connects = [4, 5, 6, 7, 8]
steps = [2, 3, 4]
querys = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
for connect in connects:
    for step in steps:
        index = dehnswpy.Index(dehnswpy.Distance_Type.Euclidean2, int(len(train[0])), int(connect),
                               int(128),
                               int(step), 1000000000)
        for i in range(len(train)):
            dehnswpy.insert(index, train[i])
        index = dehnswpy.optimize(index)
        for query in querys:
            total_time = 0
            total_hit = 0
            for i in range(len(test)):
                start = time.time()
                results = dehnswpy.query(index, test[i], len(neighbors[i]), query)
                end = time.time()
                total_time += end - start
                total_hit += verify(reference_answer, i, results)
            rf.write("{:7d}".format(total_hit))
            rf.write("    ")
            rf.write("{}".format(total_time * 1000000 / len(test)))
            rf.write("\n")
