#!/bin/bash

cmake --build build --config Release --target performence

data=fashion-mnist

LL=(8 16 32 64)
UL=(16 32 64 128)
CR=(3 4 5 6)
BM=(30 50 100)

numactl --physcpubind=0,2,4,6,8,10,24,26,28,30,32,34 --localalloc \
    ./binary/release/performence \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    "${LL[*]}" \
    "${UL[*]}" \
    "${CR[*]}" \
    "${BM[*]}"

for l in ${LL[*]}
do
    u=$(( 2 * ${l} ))
    for c in ${CR[*]}
    do
        for b in ${BM[*]}
        do
            cat result/HSG/${data}-${l}-${u}-${c}-${b}.txt | grep hit >> result/HSG/HSG-${data}.txt
        done
    done
done

# data=sift

# LL=(8 16 32 64)
# UL=(16 32 64 128)
# CR=(3 4 5 6)
# BM=(30 50 100)

# numactl --cpunodebind=0 --localalloc \
#     ./binary/release/performence \
#     ./data/${data}/train \
#     ./data/${data}/test \
#     ./data/${data}/neighbors \
#     ./data/${data}/reference_answer \
#     ${data} \
#     "${LL[*]}" \
#     "${UL[*]}" \
#     "${CR[*]}" \
#     "${BM[*]}"

# for l in ${ll[*]}
# do
#     u=$(( 2 * ${l} ))
#     for c in ${CR[*]}
#     do
#         for b in ${BM[*]}
#         do
#             cat result/HSG/${data}-${l}-${u}-${c}-${b}.txt | grep hit >> result/HSG/HSG-${data}.txt
#         done
#     done
# done

# data=gist

# LL=(8 16 32 64)
# UL=(16 32 64 128)
# CR=(3 4 5 6)
# BM=(30 50 100)

# numactl --cpunodebind=0 --localalloc \
#     ./binary/release/performence \
#     ./data/${data}/train \
#     ./data/${data}/test \
#     ./data/${data}/neighbors \
#     ./data/${data}/reference_answer \
#     ${data} \
#     "${LL[*]}" \
#     "${UL[*]}" \
#     "${CR[*]}" \
#     "${BM[*]}"

# for l in ${ll[*]}
# do
#     u=$(( 2 * ${l} ))
#     for c in ${CR[*]}
#     do
#         for b in ${BM[*]}
#         do
#             cat result/HSG/${data}-${l}-${u}-${c}-${b}.txt | grep hit >> result/HSG/HSG-${data}.txt
#         done
#     done
# done

# data=sift1B

# LL=(8 16 32 64)
# UL=(16 32 64 128)
# CR=(3 4 5 6)
# BM=(30 50 100)

# numactl --cpunodebind=0 --localalloc \
#     ./binary/release/performence \
#     ./data/${data}/bigann_base.bvecs \
#     ./data/${data}/bigann_query.bvecs \
#     ./data/${data}/gnd/idx_10M.ivecs \
#     ./data/${data}/reference_answer \
#     ${data} \
#     "${LL[*]}" \
#     "${UL[*]}" \
#     "${CR[*]}" \
#     "${BM[*]}"

# for l in ${ll[*]}
# do
#     u=$(( 2 * ${l} ))
#     for c in ${CR[*]}
#     do
#         for b in ${BM[*]}
#         do
#             cat result/HSG/${data}-${l}-${u}-${c}-${b}.txt | grep hit >> result/HSG/HSG-${data}.txt
#         done
#     done
# done
