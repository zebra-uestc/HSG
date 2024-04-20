#!/bin/bash

data=fashion-mnist
numactl --physcpubind=12-23 --localalloc ./binary/release/hnsw ./data/${data}/train ./data/${data}/test ./data/${data}/neighbors
# data=sift
# ./binary/release/hnsw ./data/${data}/train ./data/${data}/test ./data/${data}/neighbors
# data=gist
# ./binary/release/hnsw ./data/${data}/train ./data/${data}/test ./data/${data}/neighbors
