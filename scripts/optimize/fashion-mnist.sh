#!/bin/bash

cmake --build build/clang --config Release --target optimize
# cmake --build build/gcc --config Debug --target optimize

data=fashion-mnist
numactl --cpunodebind=0 --localalloc \
    ./binary/release/optimize \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    4 8 4 50 10 100
