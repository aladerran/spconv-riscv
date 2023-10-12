#include <algorithm>
#include <vector>
#include "systolic_include.h"

// Naive matmul
void slow_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < K; ++j) {
            float sum = 0.0;
            for(int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

// Gemmini matmul
void gemmini_matmul(const float *A, const float *B, float *C, int M, int N, int K,
                    tiled_matmul_type_t tiled_matmul_type){

    tiled_matmul_auto((size_t)M, (size_t)N, (size_t)K, 
                    (elem_t*)A, (elem_t*)B, 
                    NULL, C, 
                    (size_t)K, (size_t)N, (size_t)N, (size_t)N, 
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    false, false,
                    false, 0,
                    0,
                    tiled_matmul_type);
}


void scatter_cpu(const int n_in, const int n_out, const int c,
                 const float *in_feat, float *out_feat, const int *kmap,
                 const bool transpose) {
    for (int i = 0; i < n_in; i++) {
        int out_pos = kmap[2 * i + 1 - transpose];
        if (out_pos < 0) {
            continue;
        }
#pragma omp parallel for
        for (int j = 0; j < c; j++) {
            out_feat[out_pos * c + j] += in_feat[i * c + j];
        }
    }
}

void gather_cpu(const int n_k, const int n_in, const int c,
                const float *in_feat, float *out_feat, const int *kmap,
                const bool transpose) {
    for (int i = 0; i < n_k; i++) {
        int in_pos = kmap[2 * i + transpose];
        if (in_pos < 0) {
            continue;
        }
#pragma omp parallel for
        for (int j = 0; j < c; j++) {
            out_feat[i * c + j] = in_feat[in_pos * c + j];
        }
    }
}


void convolution_forward_cpu(float *in_feat, float *out_feat,
                             float *kernel, int *neighbor_map,
                             int *neighbor_offset, const bool transpose,
                             const int in_nrows, const int out_nrows,
                             const int kernel_volume, const int c, 
                             enum tiled_matmul_type_t tiled_matmul_type) {

    // Initialize output feature with zeros
    std::fill(out_feat, out_feat + out_nrows * c, 0.0f);

    int in_buffer_size =
        *std::max_element(neighbor_offset, neighbor_offset + kernel_volume);

    std::vector<float> in_buffer(in_buffer_size * c, 0.0f);
    std::vector<float> out_buffer(in_buffer_size * c, 0.0f);
    int cur_offset = 0;
    for (int i = 0; i < kernel_volume; i++) {

        if (neighbor_offset[i] == 0) {
            continue;
        }

        float *out_buffer_activated = &out_buffer[0];
        float *in_buffer_activated = &in_buffer[0];

        // gather
        gather_cpu(neighbor_offset[i], in_nrows, c,
                   in_feat, in_buffer_activated,
                   neighbor_map + cur_offset, transpose);

        // matmul
        if (tiled_matmul_type == CPU){
            slow_matmul(in_buffer_activated, kernel + i * c * c, out_buffer_activated, neighbor_offset[i], c, c);
        }
        else{
            gemmini_matmul(in_buffer_activated, kernel + i * c * c, out_buffer_activated, neighbor_offset[i], c, c, tiled_matmul_type);
        }
        // scatter
        scatter_cpu(neighbor_offset[i], out_nrows, c,
                    out_buffer_activated, out_feat,
                    neighbor_map + cur_offset, transpose);
        cur_offset += 2 * neighbor_offset[i];
    }
}