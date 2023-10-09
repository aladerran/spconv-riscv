


if [ ! -d sparsehash ]; then
    git clone --recursive https://github.com/sparsehash/sparsehash.git
fi

if [ ! -d google ]; then
    mkdir google
fi

cd sparsehash

./configure --host=riscv64-unknown-linux-gnu 
make
make install DESTDIR=../google

rm -rf sparsehash