#include <algorithm>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "systolic_include.h"

// Naive matmul
void slow_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            float sum = 0.0;
            for(int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
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

void matmul_type_dispatch(tiled_matmul_type_t tiled_matmul_type, 
                          const float* A, const float* B, float* C, 
                          int M, int N, int K) {
    switch (tiled_matmul_type) {
        case CPU:
            slow_matmul(A, B, C, M, N, K);
            break;
        case OS:
            std::cout << "Using Gemmini OS Matmul！" << std::endl;
            gemmini_matmul(A, B, C, M, N, K, OS);
            break;
        case WS:
            std::cout << "Using Gemmini WS Matmul!" << std::endl;
            gemmini_matmul(A, B, C, M, N, K, WS);
            break;
        default:
            throw std::invalid_argument("Invalid matmul type");
    }
}



void scatter_cpu(const int n_in, const int n_out, const int c,
                 const float *in_feat, float *out_feat, const int *kmap,
                 const bool transpose) {
    for (int i = 0; i < n_in; i++) {
        int out_pos = kmap[2 * i + 1 - transpose];
        if (out_pos < 0) {
            continue;
        }
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
        for (int j = 0; j < c; j++) {
            out_feat[i * c + j] = in_feat[in_pos * c + j];
        }
    }
}


void convolution_forward_cpu(const std::vector<float>& in_feat,
                             std::vector<float>& out_feat,
                             const std::vector<float>& kernel,
                             const std::vector<int>& neighbor_map,
                             const std::vector<int>& neighbor_offset,
                             const bool transpose,
                             const int in_channels,
                             const int out_channels,
                             const int in_nrows,
                             const int out_nrows,
                             const int kernel_volume,
                             tiled_matmul_type_t tiled_matmul_type){

    // Resize the out_feat vector and fill it with zeros
    out_feat.resize(out_nrows * out_channels);
    std::fill(out_feat.begin(), out_feat.end(), 0.0f);

    int in_buffer_size = 1;
    bool flag = false;
    // memory optimization
    // if (kernel_volume % 2 && out_nrows == in_nrows) {
    //     flag = true;
    //     in_buffer_size =
    //         *std::max_element(neighbor_offset.begin(),
    //                           neighbor_offset.begin() + kernel_volume / 2);
    //     in_buffer_size =
    //         std::max(in_buffer_size,
    //                  *std::max_element(
    //                      neighbor_offset.begin() + kernel_volume / 2 + 1,
    //                      neighbor_offset.begin() + kernel_volume));
    //     in_buffer_size = std::max(in_buffer_size, 1);

    //     // Perform matmul for the center kernel if conditions are met
    //     matmul_type_dispatch(tiled_matmul_type, &in_feat[0], 
    //                          &kernel[kernel_volume / 2 * in_channels * out_channels],
    //                          &out_feat[0], out_nrows, in_channels, out_channels);
    // } else {
        in_buffer_size =
            *std::max_element(neighbor_offset.begin(), neighbor_offset.begin() + kernel_volume);
    // }

    std::vector<float> in_buffer(in_buffer_size * in_channels, 0.0f);
    std::vector<float> out_buffer(in_buffer_size * out_channels, 0.0f);

    int cur_offset = 0;
    for (int i = 0; i < kernel_volume; i++) {
        if (flag && (i == kernel_volume / 2)) {
            cur_offset += 2 * neighbor_offset[i];
            continue;
        }

        if (neighbor_offset[i] == 0) {
            continue;
        }

        // gather
        gather_cpu(neighbor_offset[i], in_nrows, in_channels,
                   &in_feat[0], &in_buffer[0],
                   &neighbor_map[cur_offset], transpose);

        // matmul
        matmul_type_dispatch(tiled_matmul_type, &in_buffer[0], 
                             &kernel[i * in_channels * out_channels],
                             &out_buffer[0], neighbor_offset[i], in_channels, out_channels);

        // scatter
        scatter_cpu(neighbor_offset[i], out_nrows, out_channels,
                    &out_buffer[0], &out_feat[0],
                    &neighbor_map[cur_offset], transpose);
        cur_offset += 2 * neighbor_offset[i];
    }
}
