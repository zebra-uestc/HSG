#!/bin/bash

cmake --build build/clang --config Release --target DI
# cmake --build build/gcc --config Debug --target DI

data=fashion-mnist
numactl --cpunodebind=0 --localalloc \
    ./binary/release/DI \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    4 8 5 400 10 50 \
    ./data/${data}/delete75irrelevant.binary
