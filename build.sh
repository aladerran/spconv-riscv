#!/bin/bash
set -e

if [ ! -d build ]; then
  mkdir build
fi

# if [ ! -d libtorch ]; then
#   wget ...... -O libtorch.zip #TODO: Add pre-built Libtorch for RISC-V
#   unzip libtorch.zip
#   rm -f libtorch.zip
# fi

cd build
cmake ..
make
cd ..
