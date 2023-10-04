#!/bin/bash
set -e
# A script for mamually building Libtorch for RISC-V


# if [ ! -d pytorch ]; then
#     git clone --recursive https://github.com/pytorch/pytorch.git
# fi


# cd pytorch
# git fetch --all
# git checkout v2.0.0
# git submodule update --init --recursive 
# cp -r ../.misc/helperpurec_scalar.h third_party/sleef/src/arch/helperpurec_scalar.h

# rm -rf build && mkdir build && cd build

cd pytorch/build

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../../cmake/riscv.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DUSE_CUDA=OFF \
  -DUSE_ROCM=OFF \
  -DUSE_MKLDNN=OFF \
  -DUSE_OPENMP=OFF \
  -DUSE_DISTRIBUTED=OFF \
  -DUSE_NCCL=OFF \
  -DUSE_CUDNN=OFF \
  -DUSE_FBGEMM=OFF \
  -DUSE_QNNPACK=OFF \
  -DUSE_PYTORCH_QNNPACK=OFF \
  -DBUILD_PYTHON=OFF \
  -DBUILD_CAFFE2_OPS=OFF \
  -DBUILD_TEST=OFF \
  -DUSE_LEVELDB=OFF \
  -DUSE_LMDB=OFF 
  
make -j$(nproc)
