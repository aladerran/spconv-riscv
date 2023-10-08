# ---------------------------------------------------------------------------- #
# Add gemmini conv tests

add_compile_definitions(BAREMETAL=1)
add_compile_options(-mcmodel=medany -fno-tree-loop-distribute-patterns -fno-builtin-printf -fno-common)
add_link_options(-nostartfiles -nostdlib)

add_library(
    gemmini_rt
    "${gemmini-rocc-tests_SOURCE_DIR}/riscv-tests/benchmarks/common/syscalls.c"
	"${gemmini-rocc-tests_SOURCE_DIR}/riscv-tests/benchmarks/common/crt.S"
)
target_compile_definitions(
    gemmini_rt
    PRIVATE
    PREALLOCATE=1
	MULTITHREAD=1
	ID_STRING=
)
target_link_options(
    gemmini_rt
    INTERFACE
    "SHELL:-T ${gemmini-rocc-tests_SOURCE_DIR}/riscv-tests/benchmarks/common/test.ld"
)
target_include_directories(
    gemmini_rt
    SYSTEM PUBLIC
    "${gemmini-rocc-tests_SOURCE_DIR}/riscv-tests"
    "${gemmini-rocc-tests_SOURCE_DIR}/riscv-tests/env"
    "${gemmini-rocc-tests_SOURCE_DIR}/riscv-tests/benchmarks/common"
)

add_executable(gemmini_conv conv.c)

target_include_directories(gemmini_conv PUBLIC "${gemmini-rocc-tests_SOURCE_DIR}")
target_link_libraries(gemmini_conv PUBLIC gemmini_rt)
