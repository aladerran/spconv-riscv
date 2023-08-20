cmake_minimum_required(VERSION 3.22)

# NOTICE: this project only has any hope of compiling with the RISC-V toolchain
# provided by UCB-BAR. Thus, the included toolchain file is set by default, even
# though this is typically a bad idea.
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/cmake/riscv.cmake"
    CACHE FILEPATH "CMake toolchain file")

project(gemmini-async_sparse LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 11)

# ---------------------------------------------------------------------------- #
# Dependencies

## gemmini-rocc-tests
include(FetchContent)

FetchContent_Declare(
  gemmini-rocc-tests
  GIT_REPOSITORY https://github.com/ucb-bar/gemmini-rocc-tests.git
  GIT_TAG        6ad9474
)

FetchContent_MakeAvailable(gemmini-rocc-tests)

# ---------------------------------------------------------------------------- #


add_subdirectory(src)
add_subdirectory(test)
