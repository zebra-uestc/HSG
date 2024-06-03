#!/bin/bash

cmake --build build --config Release --target performence

data=sift1B

LL=(4 8 16 32)
UL=(8 16 32 64)
CR=(4 5 6 7)
BM=(30 50 100)

numactl --cpunodebind=1 --localalloc \
    ./binary/release/performence \
    ./data/${data}/bigann_base.bvecs \
    ./data/${data}/bigann_query.bvecs \
    ./data/${data}/gnd/idx_10M.ivecs \
    ./data/${data}/reference_answer \
    ${data} \
    "${LL[*]}" \
    "${UL[*]}" \
    "${CR[*]}" \
    "${BM[*]}"

for l in ${ll[*]}
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
