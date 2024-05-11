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
    """
    Transforms the dataset from the HDF5 format to conventional numpy format.

    If the dataset is dense, it's returned as a numpy array.
    If it's sparse, it's transformed into a list of numpy arrays, each representing a data sample.

    Args:
        dataset (h5py.Dataset): The input dataset in HDF5 format.

    Returns:
        Tuple[Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]: Tuple of training and testing data in conventional format.
    """

    for i in dataset:
        print(i, end="\n")
    print()
    for i in dataset.attrs:
        print(i, end="\n")
    print()
    print(dataset.attrs.get("distance"))
    print()
    print(len(np.array(dataset["train"])))
    print()
    print(len(np.array(dataset["train"])[0]))
    print()
    print(type(np.array(dataset["train"])[0][0]))
    print()
    print(len(np.array(dataset["test"])))
    print()
    print(len(np.array(dataset["test"])[0]))
    print()
    print(type(np.array(dataset["test"])[0][0]))
    print()
    print(len(np.array(dataset["neighbors"])))
    print()
    print(len(np.array(dataset["neighbors"])[0]))
    print()
    print(type(np.array(dataset["neighbors"])[0][0]))
    print()

    return (
        np.array(dataset["train"]),
        np.array(dataset["test"]),
        np.array(dataset["neighbors"]),
    )


f = h5py.File("sift-128-euclidean.hdf5", "r")
train, test, neighbors = dataset_transform(f)
with open("train", "wb") as file:
    file.write(struct.pack("Q", len(train)))
    print(len(train))
    file.write(struct.pack("Q", train[0].size))
    print(train[0].size)
    for i in train:
        for j in i:
            file.write(struct.pack("f", j))
file.close()
with open("test", "wb") as file:
    file.write(struct.pack("Q", len(test)))
    print(len(test))
    file.write(struct.pack("Q", test[0].size))
    print(test[0].size)
    for i in test:
        for j in i:
            file.write(struct.pack("f", j))
file.close()
with open("neighbors", "wb") as file:
    file.write(struct.pack("Q", len(neighbors)))
    print(len(neighbors))
    file.write(struct.pack("Q", neighbors[0].size))
    print(neighbors[0].size)
    for i in neighbors:
        for j in i:
            file.write(struct.pack("Q", j))
file.close()
