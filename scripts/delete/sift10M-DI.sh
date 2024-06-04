#!/bin/bash

cmake --build build --config Release --target DI
# cmake --build build --config Debug --target DI

data=sift10M
numactl --cpunodebind=0 --localalloc \
    ./binary/release/DI \
    ./data/${data}/bigann_base.bvecs \
    ./data/${data}/bigann_query.bvecs \
    ./data/${data}/gnd/idx_10M.ivecs \
    ./data/${data}/reference_answer \
    ${data} \
    16 32 6 400 100 50 \
    ./data/${data}/delete75irrelevant.binary
