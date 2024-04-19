#!/bin/bash

data=fashion-mnist
./binary/release/miluann_example ./data/${data}/train ./data/${data}/test ./data/${data}/neighbors
# data=sift
# ./binary/release/miluann_example ./data/${data}/train ./data/${data}/test ./data/${data}/neighbors
# data=gist
# ./binary/release/miluann_example ./data/${data}/train ./data/${data}/test ./data/${data}/neighbors
