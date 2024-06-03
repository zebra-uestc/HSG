#!/bin/bash

cmake --build build --config Release --target performence

data=fashion-mnist

LL=(4 8 16 32)
UL=(8 16 32 64)
CR=(4 5 6 7)
BM=(30 50 100)

numactl --physcpubind=0,2,4,6,8,10 --localalloc \
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
