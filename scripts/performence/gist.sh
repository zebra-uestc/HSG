#!/bin/bash

cmake --build build --config Release --target performence

data=gist

LL=(8 16 32)
UL=(16 32 64)
CR=(4 5 6 7)
K=100
BM=(30 50 100 200 400 800)

numactl --cpunodebind=0 --localalloc \
    ./binary/release/performence \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    ./data/${data}/reference_answer \
    ${data} \
    "${LL[*]}" \
    "${UL[*]}" \
    "${CR[*]}" \
    ${K} \
    "${BM[*]}"

for l in ${LL[*]}
do
    u=$(( 2 * ${l} ))
    for c in ${CR[*]}
    do
        for b in ${BM[*]}
        do
            cat result/HSG/${data}-${l}-${u}-${c}-${K}-${b}.txt | grep hit >> result/HSG/HSG-${data}.txt
        done
    done
done
