#!/bin/bash

if [ ! -d build ]; then
  mkdir build
fi

if [ ! -d libtorch ]; then
  wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-static-with-deps-1.10.0%2Bcpu.zip -O libtorch.zip
  unzip libtorch.zip
  rm -f libtorch.zip
fi

cd build
cmake ..
make
cd ..
