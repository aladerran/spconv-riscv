#include <google/dense_hash_map>

void hash_query_cpu(const int64_t* hash_query, const int64_t* hash_target,
                    const int64_t* idx_target, int64_t* out, const int n, const int n1) {
    google::dense_hash_map<int64_t, int64_t> hashmap;
    hashmap.set_empty_key(0);
    
    for (int idx = 0; idx < n; idx++) {
        int64_t key = hash_target[idx];
        int64_t val = idx_target[idx] + 1;
        hashmap.insert(std::make_pair(key, val));
    }

    for (int idx = 0; idx < n1; idx++) {
        int64_t key = hash_query[idx];
        google::dense_hash_map<int64_t, int64_t>::iterator iter = hashmap.find(key);
        if (iter != hashmap.end()) {
            out[idx] = iter->second;
        }
    }
}