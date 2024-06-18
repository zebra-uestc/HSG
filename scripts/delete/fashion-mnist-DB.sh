#!/bin/bash

cmake --build build/clang --config Release --target DB
# cmake --build build/gcc --config Debug --target DB

data=fashion-mnist
numactl --cpunodebind=0 --localalloc \
    ./binary/release/DB \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    4 8 5 100 10 100 \
    ./data/${data}/delete50irrelevant.binary \
    ./data/${data}/save10relevant.binary