#!/bin/bash

cmake --build build --config Release --target optimize
# cmake --build build --config Debug --target optimize

data=fashion-mnist
numactl --cpunodebind=0 --localalloc \
    ./binary/release/optimize \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    4 8 5 50 50

# data=sift
# numactl --cpunodebind=0 --localalloc \
#     ./binary/release/optimize \
#     ./data/${data}/train \
#     ./data/${data}/test \
#     ./data/${data}/neighbors \
#     ./data/${data}/reference_answer \
#     ${data} \
#     "4 8 16 24 32" \
#     "8 16 32 48 64" \
#     "3 4 5" \
#     "10 30 50"

# data=gist
# numactl --cpunodebind=0 --localalloc \
#     ./binary/release/optimize \
#     ./data/${data}/train \
#     ./data/${data}/test \
#     ./data/${data}/neighbors \
#     ./data/${data}/reference_answer \
#     ${data} \
#     "4 8 16 24 32" \
#     "8 16 32 48 64" \
#     "3 4 5 6" \
#     "10 30 50"

# data=sift1B
# numactl --cpunodebind=0 --localalloc \
#     ./binary/release/optimize \
#     ./data/${data}/bigann_base.bvecs \
#     ./data/${data}/bigann_query.bvecs \
#     ./data/${data}/gnd/idx_10M.ivecs \
#     ./data/${data}/reference_answer \
#     ${data} \
#     "4 8 16 24 32" \
#     "8 16 32 48 64" \
#     "3 4 5 6" \
#     "10 30 50"
