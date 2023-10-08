#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <vector>

const int MAX_NEIGHBORS = 8;

// Random float generation
float randomFloat(float a, float b) {
    return (b - a) * ((float)rand() / RAND_MAX) + a;
}

// Generate a randomized input feature tensor
float** generateInFeat(int n, int c) {
    float** in_feat = new float*[n];
    for (int i = 0; i < n; i++) {
        in_feat[i] = new float[c];
        for (int j = 0; j < c; j++) {
            in_feat[i][j] = randomFloat(-10.0f, 10.0f);
        }
    }
    return in_feat;
}

// Generate a randomized kernel tensor
float*** generateKernel(int k, int in_channels, int out_channels) {
    float*** kernel = new float**[k];
    for (int i = 0; i < k; i++) {
        kernel[i] = new float*[in_channels];
        for (int j = 0; j < in_channels; j++) {
            kernel[i][j] = new float[out_channels];
            for (int l = 0; l < out_channels; l++) {
                kernel[i][j][l] = randomFloat(-1.0f, 1.0f);
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