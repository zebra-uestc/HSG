#!/bin/bash

cmake --build build/clang --config Release --target optimize
# cmake --build build/gcc --config Debug --target optimize

data=gist
nohup \
    numactl --physcpubind=22 --localalloc \
    ./binary/release/optimize \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    4 8 5 50 100 50 \
    &
