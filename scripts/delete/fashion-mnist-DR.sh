#!/bin/bash

cmake --build build/clang --config Release --target DR
# cmake --build build/gcc --config Debug --target DR

data=fashion-mnist
numactl --cpunodebind=0 --localalloc \
    ./binary/release/DR \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    4 8 5 100 10 100 \
    ./data/${data}/save10relevant.binary
