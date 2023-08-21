# ---------------------------------------------------------------------------- #
# Add rpg_asynet tests

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(rpg_asynet_conv conv.cpp)

target_include_directories(rpg_asynet_conv PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(rpg_asynet_conv PUBLIC "${gemmini-rocc-tests_SOURCE_DIR}")
target_include_directories(rpg_asynet_conv PUBLIC  "${CMAKE_SOURCE_DIR}")
target_link_libraries(rpg_asynet_conv PUBLIC async_sparse_lib)