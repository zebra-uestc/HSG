#!/bin/bash

# for s in 4 8 16 24 32
# do
#     for b in 5 10 30 50
#     do
#         for p in 1 1.1 1.3 1.6 2
#         do
#             cat result/milu/${s}-${b}-${p}.txt >> result.txt
#         done
#     done
# done

for l in 4 8 16 24 32
do
    for u in 8 16 32 48 64
    do
        for c in 3 4 5
        do
            for b in 10 30 50
            do
                cat ${l}-${u}-${c}-${b}.txt | grep hit >> result.txt
            done
        done
    done
done
