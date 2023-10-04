#include "convolution_cpu.h"
#include "include/gemmini_testutils.h"

int main(){
    uint64_t start = read_cycles();
    uint64_t end = read_cycles();
    printf("torchsparse took %d cycles\n", end-start);
    return 0;
}