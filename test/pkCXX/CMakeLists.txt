# ---------------------------------------------------------------------------- #
# Add asyncSpConv tests

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(sparse_conv sparse_conv.cpp)

target_include_directories(sparse_conv PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(sparse_conv PUBLIC "${gemmini-rocc-tests_SOURCE_DIR}")
target_include_directories(sparse_conv PUBLIC  ${CMAKE_SOURCE_DIR})
target_link_libraries(sparse_conv PUBLIC async_sparse_lib)