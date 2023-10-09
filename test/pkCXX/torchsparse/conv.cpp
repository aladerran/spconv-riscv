#include "convolution_cpu.h"
#include "hashmap_cpu.h"
#include "helper.h"
#include "include/gemmini_testutils.h"
#include <cstdint>

int main() {
    
    int n = 5; 
    int c = 4; 
    int k = 3; 
    int out_c = 6;

    int8_t** in_feat = generateInFeat(n, c);
    int8_t*** kernel = generateKernel(k, c, out_c);
    int* neighbor_map = generateNeighborMap(n);
    int* neighbor_offset = generateNeighborOffset(k);

    // Print the input
    printf("in_feat:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < out_c; j++) {
            std::cout << static_cast<int>(in_feat[i][j]) << " ";  // Convert to int for printing
        }
        std::cout << std::endl;
    }

    int8_t** out_feat = new int8_t*[n];
    for (int i = 0; i < n; i++) {
        out_feat[i] = new int8_t[out_c];
    }

    int8_t** out_feat_gemmini = new int8_t*[n];
    for (int i = 0; i < n; i++) {
        out_feat_gemmini[i] = new int8_t[out_c];
    }

    uint64_t start = read_cycles();
    convolution_forward_cpu(in_feat, n, c, out_feat, n, out_c, kernel, k, c, out_c, neighbor_map, neighbor_offset, false);
    uint64_t end = read_cycles();
    printf("Torchsparse convolution_forward_cpu took %d cycles\n", end-start);

    uint64_t gemmini_start = read_cycles();
    convolution_forward_gemmini(in_feat, n, c, out_feat_gemmini, n, out_c, kernel, k, c, out_c, neighbor_map, neighbor_offset, false);
    uint64_t gemmini_end = read_cycles();
    printf("Torchsparse convolution_forward_gemmini took %d cycles\n", gemmini_end-gemmini_start);
    

    printf("out_feat_cpu:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < out_c; j++) {
            std::cout << static_cast<int>(out_feat[i][j]) << " ";  // Convert to int for printing
        }
        std::cout << std::endl;
        delete[] out_feat[i];
    }
    delete[] out_feat;

    printf("out_feat_gemmini:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < out_c; j++) {
            std::cout << static_cast<int>(out_feat_gemmini[i][j]) << " ";  // Convert to int for printing
        }
        std::cout << std::endl;
        delete[] out_feat_gemmini[i];
    }
    delete[] out_feat_gemmini;

    for (int i = 0; i < n; i++) {
        delete[] in_feat[i];
    }
    delete[] in_feat;

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < c; j++) {
            delete[] kernel[i][j];
        }
        delete[] kernel[i];
    }
    delete[] kernel;

    delete[] neighbor_map;
    delete[] neighbor_offset;

    return 0;
}
