#include <cmath>
#include <google/dense_hash_map>
#include <iostream>
#include <vector>
#include "hashmap_cpu.h"

std::vector<int64_t> hash_query_cpu(const std::vector<int64_t>& hash_query,
                                    const std::vector<int64_t>& hash_target,
                                    const std::vector<int64_t>& idx_target) {
    int n = hash_target.size();
    int n1 = hash_query.size();

    google::dense_hash_map<int64_t, int64_t> hashmap;
    hashmap.set_empty_key(0);
    std::vector<int64_t> out(n1, 0);
    
    for (int idx = 0; idx < n; idx++) {
        int64_t key = hash_target[idx];
        int64_t val = idx_target[idx] + 1;
        hashmap.insert(std::make_pair(key, val));
    }
#pragma omp parallel for
    for (int idx = 0; idx < n1; idx++) {
        int64_t key = hash_query[idx];
        auto iter = hashmap.find(key);
        if (iter != hashmap.end()) {
            out[idx] = iter->second;
        }
    }

    return out;
}
