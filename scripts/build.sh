#!/bin/bash

cmake -S . \
    -B build \
    -G"Ninja Multi-Config" \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang
