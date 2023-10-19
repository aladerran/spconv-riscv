#include "convolution_cpu.h"
#include "hash_cpu.h"
#include "hashmap_cpu.h"
#include "query_cpu.h"
#include "gemmini_testutils.h"
#include "helper.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    const int in_nrows = 512;
    const int in_channels = 256;
    const int out_nrows = 224;
    const int out_channels = 224;
    const int kernel_volume = 3;
    bool transpose = false;

    std::cout << "Randomizing input..." << std::endl;

    auto test_data_conv = generate_test_conv(in_nrows, kernel_volume, in_channels, out_channels);
    float* in_feat = std::get<0>(test_data_conv).data();
    float* kernel = std::get<1>(test_data_conv).data();
    int* neighbor_map = std::get<2>(test_data_conv).data();
    int* neighbor_offset = std::get<3>(test_data_conv).data();
    // Allocate memory for output features
    float* out_feat = new float[out_nrows * out_channels];
    float* out_feat_gemmini = new float[out_nrows * out_channels];

    std::cout << "=========Test convolution_forward_cpu begins=========" << std::endl;

    int64_t cpu_start = read_cycles();
    convolution_forward_cpu(in_feat, out_feat, kernel,
                            neighbor_map, neighbor_offset,
                            transpose, in_channels, out_channels, in_nrows, out_nrows, kernel_volume, CPU);
    int64_t cpu_end = read_cycles();
    std::cout << "Naive forward took " << cpu_end-cpu_start << " cycles" << std::endl;

    int64_t gemmini_start = read_cycles();
    convolution_forward_cpu(in_feat, out_feat_gemmini, kernel,
                            neighbor_map, neighbor_offset,
                            transpose, in_channels, out_channels, in_nrows, out_nrows, kernel_volume, WS);
    int64_t gemmini_end = read_cycles();
    std::cout << "Gemmini forward took " << gemmini_end-gemmini_start << " cycles" << std::endl;

    // if (check_featureMap(out_feat, out_feat_gemmini, out_nrows, out_channels)) {
    //     std::cout << "=========Test convolution_forward_cpu succeeds=========" << std::endl;
    // } else {
    //     std::cout << "=========Test convolution_forward_cpu fails=========" << std::endl;
    //     print_featureMap("Output feature_map", out_feat, out_nrows, out_channels);
    //     print_featureMap("Output feature_map_gemmini", out_feat_gemmini, out_nrows, out_channels);
    // }

    std::cout << "=========Test convolution_forward_cpu ends=========" << std::endl;

    delete[] out_feat;
    delete[] out_feat_gemmini;

    std::cout << "Test continues..." << std::endl;

    std::cout << "=========Test hashmap_cpu begins=========" << std::endl;

    HashTableCPU table;
    const int num_lookups = 5;
    int64_t keys_to_lookup[num_lookups] = {0, 1, 2, 3, 4};
    int64_t lookup_results[num_lookups];
    // table.insert_vals(); // TODO: implement insert() for HashTableCPU
    table.lookup_vals(keys_to_lookup, lookup_results, num_lookups);
    for (int i = 0; i < num_lookups; ++i) {
        std::cout << "Key: " << keys_to_lookup[i] << ", Value: " << lookup_results[i] << std::endl;
    }

    std::cout << "=========Test hashmap_cpu ends=========" << std::endl;

    std::cout << "Test continues..." << std::endl;

    std::cout << "=========Test generate_pc starts=========" << std::endl;

    const int size = 10;
    auto PC = generateRandomPointCloud(size);
    auto coords_rd = std::get<0>(PC).data();
    auto feats_rd = std::get<1>(PC).data();

    print_randomCoords("Random generated coords", coords_rd, size, 4);
    print_randomFeats("Random generated feats", feats_rd, size, 4);

    std::cout << "=========Test generate_pc ends=========" << std::endl;
 
    std::cout << "Test ends..." << std::endl;
 
    return 0;

}


