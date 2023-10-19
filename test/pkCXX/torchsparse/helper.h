#include <vector>
#include <random>
#include <map>
#include <tuple>
#include <cfloat>
#include <algorithm>
#include <iostream>


// Function to generate test data for convolution_forward_cpu
using ReturnType_test_conv = std::tuple<std::vector<float>, std::vector<float>, std::vector<int>, std::vector<int>>;
ReturnType_test_conv generate_test_conv(int in_nrows, int kernel_volume, int in_channels, int out_channels) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<float> in_feat(in_nrows * in_channels);
    std::vector<float> kernel(kernel_volume * in_channels * out_channels);
    std::vector<int> neighbor_map(kernel_volume * in_nrows * 2, -1);  // Initialize with -1, assuming it indicates no neighbor
    std::vector<int> neighbor_offset(kernel_volume, 0);

    for (auto& val : in_feat) val = dis(gen);
    for (auto& val : kernel) val = dis(gen);

    // Assume each row has the same number of neighbors for simplicity
    int neighbors_per_row = kernel_volume * 2;  // Adjust this value as needed
    for (int i = 0; i < in_nrows; ++i) {
        for (int j = 0; j < neighbors_per_row; ++j) {
            neighbor_map[i * neighbors_per_row + j] = static_cast<int>(dis(gen) * in_nrows);
        }
    }

    // Generate neighbor_offset based on the number of neighbors per row
    neighbor_offset[0] = 0;
    for (int i = 1; i < kernel_volume; ++i) {
        neighbor_offset[i] = neighbor_offset[i-1] + neighbors_per_row;
    }

    return std::make_tuple(in_feat, kernel, neighbor_map, neighbor_offset);
}

// Function to print out feature map outcome
void print_featureMap(const char* name, std::vector<float> feat, int nrows, int cols) {
    std::cout << "=====" << name << "=====" << std::endl;
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << feat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Helper function to check the outcome of feature map
bool check_featureMap(std::vector<float> feat1, std::vector<float> feat2, int nrows, int cols) {
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (feat1[i * cols + j] != feat2[i * cols + j]) {
                return false;
            }
        }
    }
    return true;
}


// /* Helper functions in building



// Helper functions in building */ 