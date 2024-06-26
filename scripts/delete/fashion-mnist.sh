#!/bin/bash

cmake --build build/clang --config Release --target delete
# cmake --build build/gcc --config Debug --target delete

data=fashion-mnist
numactl --cpunodebind=0 --localalloc \
    ./binary/release/delete \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    4 8 4 50 10 100 \
    ./data/${data}/delete50irrelevant.binary \
    ./data/${data}/save10relevant.binary
