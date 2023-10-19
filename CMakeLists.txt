cmake_minimum_required(VERSION 3.22)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/cmake/riscv.cmake"
    CACHE FILEPATH "CMake toolchain file")

project(sparse_conv-riscv LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 11)

# ---------------------------------------------------------------------------- #
# Dependencies

## gemmini-rocc-tests
include(FetchContent)

FetchContent_Declare(
  gemmini-rocc-tests
  GIT_REPOSITORY https://github.com/ucb-bar/gemmini-rocc-tests.git
  GIT_TAG        ae0cd82
)

FetchContent_MakeAvailable(gemmini-rocc-tests)

# ---------------------------------------------------------------------------- #


add_subdirectory(test)
