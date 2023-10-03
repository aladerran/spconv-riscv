cmake_minimum_required(VERSION 3.22)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(RISCV "$ENV{RISCV}" CACHE PATH "Path to RISCV toolchain")

set(CMAKE_SYSROOT "${RISCV}/sysroot")
set(CMAKE_CROSSCOMPILING_EMULATOR "${RISCV}/bin/spike" --extension=gemmini)

set(CMAKE_C_COMPILER "${RISCV}/bin/riscv64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${RISCV}/bin/riscv64-unknown-linux-gnu-g++")
set(CMAKE_CXX_FLAGS "-march=rv64imafdc -mabi=lp64d")