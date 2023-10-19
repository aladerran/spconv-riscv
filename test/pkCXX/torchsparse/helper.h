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
    std::vector<int> neighbor_map(kernel_volume * in_nrows * 2); 
    std::vector<int> neighbor_offset(kernel_volume);

    for (auto& val : in_feat) val = dis(gen);
    for (auto& val : kernel) val = dis(gen);

    // For simplicity, assume each input feature has one corresponding output feature,
    // and each position in the kernel has one neighbor.
    for (int i = 0; i < in_nrows; ++i) {
        for (int k = 0; k < kernel_volume; ++k) {
            int index = (i * kernel_volume + k) * 2;
            neighbor_map[index] = i;  // input feature index
            neighbor_map[index + 1] = i;  // output feature index (same as input for simplicity)
        }
    }
    std::fill(neighbor_offset.begin(), neighbor_offset.end(), 1);  // each kernel position has one neighbor

    return std::make_tuple(in_feat, kernel, neighbor_map, neighbor_offset);
}

// Function to print out feature map outcome
void print_featureMap(const char* name, float* feat, int nrows, int cols) {
    std::cout << "=====" << name << "=====" << std::endl;
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << feat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Function to check the outcome of feature map
bool check_featureMap(float* feat1, float* feat2, int nrows, int cols) {
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (feat1[i * cols + j] != feat2[i * cols + j]) {
                return false;
            }
        }
    }
    return true;
}

// Function to check  the outcome of hash
bool check_hash(const int64_t* arr1, const int64_t* arr2, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

// /* Helper functions in building

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <cmath>


// Functions to generate point-cloud data
struct Point {
    int x, y, z;
};

struct HashFunction {
    std::size_t operator()(const Point& p) const {
        return std::hash<int>()(p.x) ^ std::hash<int>()(p.y) ^ std::hash<int>()(p.z);
    }
};

bool operator==(const Point& p1, const Point& p2) {
    return p1.x == p2.x && p1.y == p2.y && p1.z == p2.z;
}

std::vector<Point> sparseQuantize(const std::vector<Point>& coords, float voxelSize) {
    std::unordered_set<Point, HashFunction> uniqueCoords;
    for (const auto& coord : coords) {
        Point quantizedCoord{
            static_cast<int>(std::floor(coord.x / voxelSize)),
            static_cast<int>(std::floor(coord.y / voxelSize)),
            static_cast<int>(std::floor(coord.z / voxelSize))
        };
        uniqueCoords.insert(quantizedCoord);
    }

    std::vector<Point> result(uniqueCoords.begin(), uniqueCoords.end());
    return result;
}

std::vector<int> append_batchInfo(const std::vector<int>& originalCoords) {
    std::vector<int> newCoords(originalCoords.size() / 3 * 4, 0);

    for (size_t i = 0; i < originalCoords.size() / 3; ++i) {
        newCoords[i * 4] = originalCoords[i * 3];        
        newCoords[i * 4 + 1] = originalCoords[i * 3 + 1]; 
        newCoords[i * 4 + 2] = originalCoords[i * 3 + 2]; 
        // Fourth dimension is already initialized to 0
    }

    return newCoords;
}

std::pair<std::vector<int>, std::vector<float>> generateRandomPointCloud(int size = 10000, float voxelSize = 0.2) {
    std::vector<int> coords(size * 3);  // Flatten to 1D vector
    std::vector<float> feats(size * 4);  // Flatten to 1D vector

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < 3; ++j) {
            coords[i * 3 + j] = static_cast<int>(rand() % 100);  // Random coordinates
            feats[i * 4 + j] = static_cast<float>(rand()) / RAND_MAX * 10.0f;  // Random features
        }
        feats[i * 4 + 3] = static_cast<float>(rand()) / RAND_MAX * 10.0f;  // Random features
    }

    std::vector<Point> pointCoords(size);
    for (int i = 0; i < size; ++i) {
        pointCoords[i] = {coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]};
    }

    std::vector<Point> uniqueCoords = sparseQuantize(pointCoords, voxelSize);

    std::vector<int> resultCoords(uniqueCoords.size() * 3);
    for (size_t i = 0; i < uniqueCoords.size(); ++i) {
        resultCoords[i * 3] = uniqueCoords[i].x;
        resultCoords[i * 3 + 1] = uniqueCoords[i].y;
        resultCoords[i * 3 + 2] = uniqueCoords[i].z;
    }

    return {append_batchInfo(resultCoords), feats};
}

// Functions to print out point-cloud data
void print_randomCoords(const char* name, int* feat, int nrows, int cols) {
    std::cout << "=====" << name << "=====" << std::endl;
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << feat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void print_randomFeats(const char* name, float* feat, int nrows, int cols) {
    std::cout << "=====" << name << "=====" << std::endl;
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << feat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}


// Helper functions in building */ 