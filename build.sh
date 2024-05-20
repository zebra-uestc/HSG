#!/bin/bash

echo ">>>>>>>>>>>>>>>>>>>>"
echo "building \"release\""
echo
cmake -S . -B build -G"Ninja Multi-Config" -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_BUILD_TYPE=Release
echo
echo "build 'release' done"
echo ">>>>>>>>>>>>>>>>>>>>"

# echo "building \"debug\""
# cmake -S . -B debug -G"Ninja Multi-Config" -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Debug
# echo "build 'debug' done"