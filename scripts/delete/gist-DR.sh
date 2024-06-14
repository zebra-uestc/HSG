#!/bin/bash

cmake --build build/clang --config Release --target DR
# cmake --build build/gcc --config Debug --target DR

data=gist
numactl --cpunodebind=0 --localalloc \
    ./binary/release/DR \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    16 32 6 400 10 100 \
    ./data/${data}/save10relevant.binary
