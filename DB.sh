#!/bin/bash

cmake --build build --config Release --target DB
# cmake --build build --config Debug --target DB

data=fashion-mnist
numactl --cpunodebind=0 --localalloc \
    ./binary/release/DB \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    4 8 5 50 50 \
    ./data/${data}/delete75irrelevant.binary \
    ./data/${data}/save10relevant.binary

# data=sift1B
# numactl --cpunodebind=0 --localalloc \
#     ./binary/release/DB \
#     ./data/${data}/bigann_base.bvecs \
#     ./data/${data}/bigann_query.bvecs \
#     ./data/${data}/gnd/idx_10M.ivecs \
#     ./data/${data}/reference_answer \
#     ${data} \
#     "4 8 16 24 32" \
#     "8 16 32 48 64" \
#     "3 4 5 6" \
#     "10 30 50" \
#     ./data/${data}/delete75irrelevant.binary \
#     ./data/${data}/save10relevant.binary
