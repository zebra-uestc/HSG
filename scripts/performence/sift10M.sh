#!/bin/bash

cmake --build build/clang --config Release --target performence

data=sift10M

LL=(8 16 32)
UL=(16 32 64)
CR=(4 5 6 7)
K=100
BM=(400 800)

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
    "${BM[*]}" \
    ${K}


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
