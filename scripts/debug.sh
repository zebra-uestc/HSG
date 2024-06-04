#!/bin/bash

cmake --build build --config Release
# cmake --build build --config Debug

data=fashion-mnist
numactl --cpunodebind=0 --localalloc \
    ./binary/release/debug \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    4 8 4 10 100 10
