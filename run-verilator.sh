#!/bin/bash
set -e

cd ~/chipyard/sims/verilator/
echo 
echo ============ running torchsparse conv ============
./simulator-chipyard-CustomGemminiSoCConfig -c pk ~/sparse_conv-riscv/build/test/pkCXX/torchsparse/torchsparse_conv
echo 
echo ============ cycle accurate info generated ============
