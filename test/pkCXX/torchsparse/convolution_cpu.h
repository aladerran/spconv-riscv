#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring> 
#include "include/gemmini_testutils.h"

void slow_matmul(float** A, float** B, float** C, int rowsA, int colsA, int rowsB, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            C[i][j] = 0.0f;
            for (int k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void gemmini_matmul(float** A, float** B, float** C, int rowsA, int colsA, int rowsB, int colsB) {
    // elem_t flat_A[rowsA][colsA] row_align(1);
    // elem_t flat_B[rowsB][colsB] row_align(1);
    // elem_t flat_C[rowsA][colsB] row_align(1);

    // for(int i = 0; i < rowsA; ++i) {
    //     for(int j = 0; j < colsA; ++j) {
    //         flat_A[i][j] = (elem_t) A[i][j];
    //     }
    // }

    // for(int i = 0; i < rowsB; ++i) {
    //     for(int j = 0; j < colsB; ++j) {
    //         flat_B[i][j] = (elem_t) B[i][j];
    //     }
    // }

    elem_t (*flat_A)[colsA] = (elem_t (*)[colsA]) A[0];
    elem_t (*flat_B)[colsB] = (elem_t (*)[colsB]) B[0];
    elem_t flat_C[rowsA][colsB] row_align(1);


    tiled_matmul_auto(rowsA, colsB, colsA, 
                    (elem_t*)A, (elem_t*)B, 
                    NULL, (elem_t*)flat_C, 
                    colsA, colsB, colsB, colsB, 
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    false, false,
                    false, 0,
                    0,
                    CPU);




    for(int i = 0; i < rowsA; i++) {
        for(int j = 0; j < colsB; j++) {
            C[i][j] = (float) flat_C[i][j];
        }
    }

}

void scatter_cpu(const int n_in, const int n_out, const int c,
                 float** in_feat, float** out_feat, const int* kmap,
                 const bool transpose) {
    for (int i = 0; i < n_in; i++) {
        int out_pos = kmap[2 * i + 1 - transpose];
        if (out_pos < 0) {
            continue;
        }
        for (int j = 0; j < c; j++) {
            out_feat[out_pos][j] += in_feat[i][j];
        }
    }
}

void gather_cpu(const int n_k, const int n_in, const int c,
                float** in_feat, float** out_feat, const int* kmap,
                const bool transpose) {
    for (int i = 0; i < n_k; i++) {
        int in_pos = kmap[2 * i + transpose];
        if (in_pos < 0) {
            continue;
        }
        for (int j = 0; j < c; j++) {
            out_feat[i][j] = in_feat[in_pos][j];
        }
    }
}

void convolution_forward_cpu(float** in_feat, int numRowsIn, int numColsIn,
                             float** out_feat, int numRowsOut, int numColsOut,
                             float*** kernel, int kernelVolume, int kernelRows, int kernelCols,
                             int* neighbor_map, int* neighbor_offset, const bool transpose) {
    if (numColsIn != kernelRows) {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }

    for (int i = 0; i < numRowsOut; i++) {
        for (int j = 0; j < numColsOut; j++) {
            out_feat[i][j] = 0.0f;
        }
    }

    int in_buffer_size = neighbor_offset[kernelVolume - 1];
    float** in_buffer = new float*[in_buffer_size];
    for (int i = 0; i < in_buffer_size; i++) {
        in_buffer[i] = new float[numColsIn];
    }

    float** out_buffer = new float*[in_buffer_size];
    for (int i = 0; i < in_buffer_size; i++) {
        out_buffer[i] = new float[numColsOut];
    }

    int cur_offset = 0;
    for (int i = 0; i < kernelVolume; i++) {
        if (neighbor_offset[i] == 0) {
            continue;
        }

        gather_cpu(neighbor_offset[i], numRowsIn, numColsIn,
                   in_feat, in_buffer, neighbor_map + cur_offset, transpose);
        slow_matmul(in_buffer, kernel[i], out_buffer, neighbor_offset[i], numColsIn, kernelRows, kernelCols);
        scatter_cpu(neighbor_offset[i], numRowsOut, numColsOut,
                    out_buffer, out_feat, neighbor_map + cur_offset, transpose);

        cur_offset += 2 * neighbor_offset[i];
    }

    for (int i = 0; i < in_buffer_size; i++) {
        delete[] in_buffer[i];
        delete[] out_buffer[i];
    }
    delete[] in_buffer;
    delete[] out_buffer;
}

void convolution_forward_gemmini(float** in_feat, int numRowsIn, int numColsIn,
                             float** out_feat, int numRowsOut, int numColsOut,
                             float*** kernel, int kernelVolume, int kernelRows, int kernelCols,
                             int* neighbor_map, int* neighbor_offset, const bool transpose) {
    if (numColsIn != kernelRows) {
        throw std::invalid_argument("Input feature size and kernel size mismatch");
    }

    for (int i = 0; i < numRowsOut; i++) {
        for (int j = 0; j < numColsOut; j++) {
            out_feat[i][j] = 0.0f;
        }
    }

    int in_buffer_size = neighbor_offset[kernelVolume - 1];
    float** in_buffer = new float*[in_buffer_size];
    for (int i = 0; i < in_buffer_size; i++) {
        in_buffer[i] = new float[numColsIn];
    }

    float** out_buffer = new float*[in_buffer_size];
    for (int i = 0; i < in_buffer_size; i++) {
        out_buffer[i] = new float[numColsOut];
    }

    int cur_offset = 0;
    for (int i = 0; i < kernelVolume; i++) {
        if (neighbor_offset[i] == 0) {
            continue;
        }

        gather_cpu(neighbor_offset[i], numRowsIn, numColsIn,
                   in_feat, in_buffer, neighbor_map + cur_offset, transpose);

        gemmini_matmul(in_buffer, kernel[i], out_buffer, neighbor_offset[i], numColsIn, kernelRows, kernelCols);
        
        scatter_cpu(neighbor_offset[i], numRowsOut, numColsOut,
                    out_buffer, out_feat, neighbor_map + cur_offset, transpose);

        cur_offset += 2 * neighbor_offset[i];
    }

    for (int i = 0; i < in_buffer_size; i++) {
        delete[] in_buffer[i];
        delete[] out_buffer[i];
    }
    delete[] in_buffer;
    delete[] out_buffer;
}