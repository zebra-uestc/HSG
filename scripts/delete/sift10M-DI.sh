#!/bin/bash

cmake --build build/clang --config Release --target DI
# cmake --build build/gcc --config Debug --target DI

data=sift10M
numactl --cpunodebind=1 --localalloc \
    ./binary/release/DI \
    ./data/${data}/bigann_base.bvecs \
    ./data/${data}/bigann_query.bvecs \
    ./data/${data}/gnd/idx_10M.ivecs \
    ./data/${data}/reference_answer \
    ${data} \
    16 32 6 800 10 100 \
    ./data/${data}/delete50irrelevant.binary