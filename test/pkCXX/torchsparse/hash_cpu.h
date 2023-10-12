#include <vector>

void cpu_hash_wrapper(int N, const int* data, int64_t* out) {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        uint64_t hash = 14695981039346656037UL;
        for (int j = 0; j < 4; j++) {
            hash ^= (unsigned int)data[4 * i + j];
            hash *= 1099511628211UL;
        }
        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
        out[i] = hash;
    }
}

void cpu_kernel_hash_wrapper(int N, int K, const int* data,
                             const int* kernel_offset, int64_t* out) {
    for (int k = 0; k < K; k++) {
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            int cur_coord[4];
            for (int j = 0; j < 3; j++) {
                cur_coord[j] = data[i * 4 + j] + kernel_offset[k * 3 + j];
            }
            cur_coord[3] = data[i * 4 + 3];
            uint64_t hash = 14695981039346656037UL;
            for (int j = 0; j < 4; j++) {
                hash ^= (unsigned int)cur_coord[j];
                hash *= 1099511628211UL;
            }
            hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
            out[k * N + i] = hash;
        }
    }
}

std::vector<int64_t> hash_cpu(const std::vector<int>& idx) {
    int N = idx.size();
    std::vector<int64_t> out(N);
    cpu_hash_wrapper(N, idx.data(), out.data());
    return out;
}

std::vector<int64_t> kernel_hash_cpu(const std::vector<int>& idx,
                                     const std::vector<int>& kernel_offset) {
    int N = idx.size();
    int K = kernel_offset.size() / 3;
    std::vector<int64_t> out(K * N);
    cpu_kernel_hash_wrapper(N, K, idx.data(), kernel_offset.data(), out.data());
    return out;
}