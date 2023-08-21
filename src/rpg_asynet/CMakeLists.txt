# ---------------------------------------------------------------------------- #
# Async Sparse Conv Library

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_library(
    async_sparse_lib
    async_sparse_conv2d.cpp
    async_sparse_conv2d.h
    rulebook.h
)

target_include_directories(async_sparse_lib PUBLIC ${EIGEN3_INCLUDE_DIR})