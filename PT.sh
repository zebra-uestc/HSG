#!/bin/bash

data=fashion-mnist
numactl --cpunodebind=1 --localalloc \
    ./binary/release/PT \
    ./data/${data}/train \
    ./data/${data}/test \
    ./data/${data}/neighbors \
    "4 8 16 24 32" \
    "8 16 32 48 64" \
    "3 4 5" \
    "10 30 50"
