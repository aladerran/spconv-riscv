if [ ! -d pytorch ]; then
    git clone --recursive https://github.com/pytorch/pytorch.git
fi

cd pytorch

if [ ! -d build ]; then
  mkdir build
fi
cd build
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/riscv.cmake \
    -DBUILD_SHARED_LIBS=OFF