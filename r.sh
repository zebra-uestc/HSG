#!/bin/bash

data=fashion-mnist
numactl --physcpubind=0-11 --localalloc ./binary/release/test ./data/${data}/train ./data/${data}/test ./data/${data}/neighbors
# data=sift
# numactl --physcpubind=0-11 --localalloc ./binary/release/test ./data/${data}/train ./data/${data}/test ./data/${data}/neighbors
# data=gist
# numactl --physcpubind=0-11 --localalloc ./binary/release/test ./data/${data}/train ./data/${data}/test ./data/${data}/neighbors
