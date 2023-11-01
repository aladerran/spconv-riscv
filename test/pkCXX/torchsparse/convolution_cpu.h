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


void convolution_forward_cpu(const float* in_feat,
                             float* out_feat,
                             const float* kernel,
                             const int* neighbor_map,
                             const int* neighbor_offset,
                             const bool transpose,
                             const int in_channels,
                             const int out_channels,
                             const int in_nrows,
                             const int out_nrows,
                             const int kernel_volume,
                             tiled_matmul_type_t tiled_matmul_type) {
    // Initialize output features to zero
    std::fill(out_feat, out_feat + out_nrows * out_channels, 0);

    int in_buffer_size = 1;
    bool flag = false;

    // Determine buffer size for memory optimization
    if (kernel_volume % 2 && out_nrows == in_nrows) {
        flag = true;
        in_buffer_size =
            *std::max_element(neighbor_offset,
                              neighbor_offset + kernel_volume / 2);
        in_buffer_size =
            std::max(in_buffer_size,
                     *std::max_element(
                         neighbor_offset + kernel_volume / 2 + 1,
                         neighbor_offset + kernel_volume));
        in_buffer_size = std::max(in_buffer_size, 1);

        // Perform initial matrix multiplication
        matmul_type_dispatch(tiled_matmul_type,
                             in_feat,
                             kernel + (kernel_volume / 2) * in_channels * out_channels,
                             out_feat,
                             in_nrows,
                             out_channels,
                             in_channels);
    } else {
        in_buffer_size =
            *std::max_element(neighbor_offset,
                              neighbor_offset + kernel_volume);
    }

    float* in_buffer = new float[in_buffer_size * in_channels]();
    float* out_buffer = new float[in_buffer_size * out_channels]();
    int cur_offset = 0;

    for (int i = 0; i < kernel_volume; i++) {

        std::fill(out_buffer, out_buffer + in_buffer_size * out_channels, 0);

        if (flag && (i == kernel_volume / 2)) {
            cur_offset += 2 * neighbor_offset[i];
            continue;
        }

        if (neighbor_offset[i] == 0) {
            continue;
        }

        // Gather
        gather_cpu(neighbor_offset[i], in_nrows, in_channels,
                   in_feat, in_buffer,
                   neighbor_map + cur_offset, transpose);

        // Matrix multiplication
        matmul_type_dispatch(tiled_matmul_type,
                             in_buffer,
                             kernel + i * in_channels * out_channels,
                             out_buffer,
                             neighbor_offset[i],
                             out_channels,
                             in_channels);

        // Scatter
        scatter_cpu(neighbor_offset[i], out_nrows, out_channels,
                    out_buffer,
                    out_feat,
                    neighbor_map + cur_offset, transpose);
        cur_offset += 2 * neighbor_offset[i];
    }

    delete[] in_buffer;
    delete[] out_buffer;
}
