#!/bin/bash

cmake --build build --config Release --target optimize
# cmake --build build --config Debug --target optimize

data=fashion-mnist
numactl --physcpubind=23 --localalloc \
    ./binary/release/optimize \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    4 8 5 400 100 100
