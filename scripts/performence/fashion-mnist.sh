#!/bin/bash

cmake --build build/clang --config Release --target performence

data=fashion-mnist

LL=(4 8 16)
UL=(8 16 32)
CR=(4 5 6)
BM=(30 50 100)
K=10

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
    "${BM[*]}" \
    ${K}


for l in ${LL[*]}
do
    for c in ${CR[*]}
    do
        for b in ${BM[*]}
        do
            cat result/HSG/${data}-${l}-${c}-${b}.txt | grep hit >> result/HSG/HSG-${data}.txt
        done
    done
done
