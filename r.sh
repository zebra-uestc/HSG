#!/bin/bash

data=fashion-mnist
numactl --cpunodebind=1 --localalloc ./binary/release/test ./data/${data}/train ./data/${data}/test ./data/${data}/neighbors
# data=sift
# numactl --cpunodebind=1 --localalloc ./binary/release/test ./data/${data}/train ./data/${data}/test ./data/${data}/neighbors
# data=gist
# numactl --cpunodebind=1 --localalloc ./binary/release/test ./data/${data}/train ./data/${data}/test ./data/${data}/neighbors
