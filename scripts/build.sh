#!/bin/bash

cmake -S . \
    -B build/clang \
    -G"Ninja Multi-Config" \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang

cmake -S . \
    -B build/gcc \
    -G"Ninja Multi-Config" \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc
