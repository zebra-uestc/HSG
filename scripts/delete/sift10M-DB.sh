#!/bin/bash

cmake --build build/clang --config Release --target DB
# cmake --build build/gcc --config Debug --target DB

data=sift10M
numactl --cpunodebind=0 --localalloc \
    ./binary/release/DB \
    ./data/${data}/bigann_base.bvecs \
    ./data/${data}/bigann_query.bvecs \
    ./data/${data}/gnd/idx_10M.ivecs \
    ./data/${data}/reference_answer \
    ${data} \
    16 32 6 400 100 50 \
    ./data/${data}/delete75irrelevant.binary \
    ./data/${data}/save10relevant.binary
