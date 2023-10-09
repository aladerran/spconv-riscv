#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cstdint>

const int MAX_NEIGHBORS = 8;

// Random int8_t generation
int8_t randomInt8(int8_t a, int8_t b) {
    return a + rand() % (b - a + 1);
}

// Generate a randomized input feature tensor
int8_t** generateInFeat(int n, int c) {
    int8_t** in_feat = new int8_t*[n];
    for (int i = 0; i < n; i++) {
        in_feat[i] = new int8_t[c];
        for (int j = 0; j < c; j++) {
            in_feat[i][j] = randomInt8(-128, 127);  // full int8_t range
        }
    }
    return in_feat;
}

// Generate a randomized kernel tensor
int8_t*** generateKernel(int k, int in_channels, int out_channels) {
    int8_t*** kernel = new int8_t**[k];
    for (int i = 0; i < k; i++) {
        kernel[i] = new int8_t*[in_channels];
        for (int j = 0; j < in_channels; j++) {
            kernel[i][j] = new int8_t[out_channels];
            for (int l = 0; l < out_channels; l++) {
                kernel[i][j][l] = randomInt8(-128, 127);  // full int8_t range
            }
        }
    }
    return kernel;
}

int* generateNeighborMap(int n) {
    int* neighbor_map = new int[2 * MAX_NEIGHBORS];
    for (int i = 0; i < 2 * MAX_NEIGHBORS; i++) {
        neighbor_map[i] = rand() % n;  // Ensure we are always within the range [0, n)
    }
    return neighbor_map;
}

int* generateNeighborOffset(int k) {
    int* neighbor_offset = new int[k];
    for (int i = 0; i < k; i++) {
        neighbor_offset[i] = rand() % k;  // Make sure the offset is within [0, k)
    }
    return neighbor_offset;
}
