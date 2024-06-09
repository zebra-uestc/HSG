#!/bin/bash

cmake --build build/clang --config Release --target optimize
# cmake --build build/gcc --config Debug --target optimize

data=sift10M
nohup \
    numactl --physcpubind=23 --localalloc \
    ./binary/release/optimize \
    ./data/${data}/bigann_base.bvecs \
    ./data/${data}/bigann_query.bvecs \
    ./data/${data}/gnd/idx_10M.ivecs \
    ./data/${data}/reference_answer \
    ${data} \
    4 8 5 50 100 50 \
    &
