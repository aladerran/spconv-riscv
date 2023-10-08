#include "convolution_cpu.h"
#include "helper.h"
#include "include/gemmini_testutils.h"

int main() {
    
    int n = 5; 
    int c = 4; 
    int k = 3; 
    int out_c = 6;

    float** in_feat = generateInFeat(n, c);
    float*** kernel = generateKernel(k, c, out_c);
    int* neighbor_map = generateNeighborMap(n);
    int* neighbor_offset = generateNeighborOffset(k);

    // Print the input
    printf("in_feat:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < out_c; j++) {
            std::cout << in_feat[i][j] << " ";
        }
        std::cout << std::endl;
    }

    float** out_feat = new float*[n];
    for (int i = 0; i < n; i++) {
        out_feat[i] = new float[out_c];
    }

    float** out_feat_gemmini = new float*[n];
    for (int i = 0; i < n; i++) {
        out_feat_gemmini[i] = new float[out_c];
    }

    uint64_t start = read_cycles();
    convolution_forward_cpu(in_feat, n, c, out_feat, n, out_c, kernel, k, c, out_c, neighbor_map, neighbor_offset, false);
    uint64_t end = read_cycles();
    printf("Torchsparse convolution_forward_cpu took %d cycles\n", end-start);

    uint64_t gemmini_start = read_cycles();
    convolution_forward_gemmini(in_feat, n, c, out_feat_gemmini, n, out_c, kernel, k, c, out_c, neighbor_map, neighbor_offset, false);
    uint64_t gemmini_end = read_cycles();
    printf("Torchsparse convolution_forward_gemmini took %d cycles\n", gemmini_end-gemmini_start);
    

    printf("out_feat shape:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < out_c; j++) {
            std::cout << out_feat[i][j] << " ";
        }
        std::cout << std::endl;
        delete[] out_feat[i];
    }
    delete[] out_feat;

    printf("out_feat_gemmini shape:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < out_c; j++) {
            std::cout << out_feat_gemmini[i][j] << " ";
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
    
    
    // uint64_t start = read_cycles();

    // uint64_t end = read_cycles();

    // printf("torchsparse took %d cycles\n", end-start);
