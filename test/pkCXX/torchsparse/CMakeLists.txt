# ---------------------------------------------------------------------------- #
# TorchSparse Library

# list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/libtorch)
# find_package(Torch REQUIRED)

# find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_executable(
    torchsparse_conv
    conv.cpp
)

# target_include_directories(torchsparse_conv PUBLIC ${Python3_INCLUDE_DIRS})
# target_include_directories(torchsparse_conv PUBLIC ${TORCH_INCLUDE_DIRS}) #TODO: Add Libtorch for RISC-V
# target_link_libraries(torchsparse_conv PUBLIC ${TORCH_LIBRARIES})
target_include_directories(torchsparse_conv PUBLIC "${gemmini-rocc-tests_SOURCE_DIR}")