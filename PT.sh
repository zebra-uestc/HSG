#!/bin/bash

cmake --build build --config Release --target PT

data=fashion-mnist
numactl --cpunodebind=0 --localalloc \
    ./binary/release/PT \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    "4 8 16 24 32" \
    "8 16 32 48 64" \
    "3 4 5 6" \
    "10 30 50"

for l in 4 8 16 24 32
do
    u=$(( 2 * ${l} ))
    for c in 3 4 5 6
    do
        for b in 10 30 50
        do
            cat result/HSG/${data}-${l}-${u}-${c}-${b}.txt | grep hit >> result/HSG/HSG-${data}.txt
        done
    done
done

# data=sift
# numactl --cpunodebind=0 --localalloc \
#     ./binary/release/PT \
#     ./data/${data}/train \
#     ./data/${data}/test \
#     ./data/${data}/neighbors \
#     ./data/${data}/reference_answer \
#     ${data} \
#     "4 8 16 24 32" \
#     "8 16 32 48 64" \
#     "3 4 5" \
#     "10 30 50"

# for l in 4 8 16 24 32
# do
#     u=$(( 2 * ${l} ))
#     for c in 3 4 5 6
#     do
#         for b in 10 30 50
#         do
#             cat result/HSG/${data}-${l}-${u}-${c}-${b}.txt | grep hit >> result/HSG/HSG-${data}.txt
#         done
#     done
# done

# data=gist
# numactl --cpunodebind=0 --localalloc \
#     ./binary/release/PT \
#     ./data/${data}/train \
#     ./data/${data}/test \
#     ./data/${data}/neighbors \
#     ./data/${data}/reference_answer \
#     ${data} \
#     "4 8 16 24 32" \
#     "8 16 32 48 64" \
#     "3 4 5 6" \
#     "10 30 50"

# for l in 4 8 16 24 32
# do
#     u=$(( 2 * ${l} ))

#     for c in 3 4 5 6
#     do
#         for b in 10 30 50
#         do
#             cat result/HSG/${data}-${l}-${u}-${c}-${b}.txt | grep hit >> result/HSG/HSG-${data}.txt
#         done
#     done
# done

# data=sift1B
# numactl --cpunodebind=0 --localalloc \
#     ./binary/release/PT \
#     ./data/${data}/bigann_base.bvecs \
#     ./data/${data}/bigann_query.bvecs \
#     ./data/${data}/gnd/idx_10M.ivecs \
#     ./data/${data}/reference_answer \
#     ${data} \
#     "4 8 16 24 32" \
#     "8 16 32 48 64" \
#     "3 4 5 6" \
#     "10 30 50"

# for l in 4 8 16 24 32
# do
#     u=$(( 2 * ${l} ))

#     for c in 3 4 5 6
#     do
#         for b in 10 30 50
#         do
#             cat result/HSG/${data}-${l}-${u}-${c}-${b}.txt | grep hit >> result/HSG/HSG-${data}.txt
#         done
#     done
# done
