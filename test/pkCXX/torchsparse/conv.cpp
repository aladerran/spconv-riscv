#include "convolution_cpu.h"
#include "hashmap_cpu.h"
#include "hash_cpu.h"
#include "query_cpu.h"
#include "gemmini_testutils.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    const int in_nrows = 24;
    const int out_nrows = 24;
    const int kernel_volume = 3;
    const int c = 3;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<float> in_feat(in_nrows * c);
    std::vector<float> kernel(kernel_volume * c * c);
    std::vector<int> neighbor_map(kernel_volume * in_nrows * 2);
    std::vector<int> neighbor_offset(kernel_volume);

    for (auto& val : in_feat) val = dis(gen);
    for (auto& val : kernel) val = dis(gen);
    for (auto& val : neighbor_map) val = dis(gen) * in_nrows;
    for (auto& val : neighbor_offset) val = dis(gen) * in_nrows;

    std::vector<float> out_feat(out_nrows * c);
    std::vector<float> out_feat_gemmini(out_nrows * c);

    bool transpose = false;

    int64_t cpu_start = read_cycles();
    convolution_forward_cpu(in_feat.data(), out_feat.data(), kernel.data(),
                            neighbor_map.data(), neighbor_offset.data(),
                            transpose, in_nrows, out_nrows, kernel_volume, c, CPU);
    int64_t cpu_end = read_cycles();
    std::cout << "Naive forward took " << cpu_end-cpu_start << " cycles" << std::endl;

    int64_t gemmini_start = read_cycles();
    convolution_forward_cpu(in_feat.data(), out_feat_gemmini.data(), kernel.data(),
                            neighbor_map.data(), neighbor_offset.data(),
                            transpose, in_nrows, out_nrows, kernel_volume, c, WS);            
    int64_t gemmini_end = read_cycles();
    std::cout << "Gemmini forward took " << gemmini_end-gemmini_start << " cycles" << std::endl;


    std::cout << "=====Output feature_map=====" << std::endl;
    for (int i = 0; i < out_nrows; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << out_feat[i * c + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "=====Output feature_map_gemmini=====" << std::endl;
    for (int i = 0; i < out_nrows; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << out_feat_gemmini[i * c + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}



// int main() {
//     // Create an instance of HashTableCPU
//     HashTableCPU table;

//     // Define some keys and values to insert
//     const int num_inserts = 10;
//     int64_t keys_to_insert[num_inserts] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//     int64_t values_to_insert[num_inserts] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

//     // Insert the keys and values
//     table.insert_vals(keys_to_insert, values_to_insert, num_inserts);

//     // Define some keys to look up
//     const int num_lookups = 5;
//     int64_t keys_to_lookup[num_lookups] = {3, 4, 5, 11, 12};  // Includes some keys not present in the table
//     int64_t lookup_results[num_lookups];

//     // Look up the keys
//     table.lookup_vals(keys_to_lookup, lookup_results, num_lookups);

//     // Print the results
//     for (int i = 0; i < num_lookups; ++i) {
//         std::cout << "Key: " << keys_to_lookup[i] << ", Value: " << lookup_results[i] << std::endl;
//     }

//     return 0;
// }
