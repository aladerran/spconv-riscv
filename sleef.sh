#!/bin/bash
set -e
# A script for cross-compiling SLEEF for RISC-V

cd pytorch/third_party/sleef 

rm -rf build && mkdir build && cd build
cmake -GNinja -DBUILD_TESTS=OFF ..
ninja
cd ..
rm -rf sleef && mkdir sleef && cd sleef
cmake -GNinja -DCMAKE_TOOLCHAIN_FILE=../../../../cmake/riscv.cmake -DBUILD_TESTS=OFF -DNATIVE_BUILD_DIR=`pwd`/../build ..
ninja

