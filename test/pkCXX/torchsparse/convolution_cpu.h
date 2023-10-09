#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring> 
#include "include/gemmini_testutils.h"

void slow_matmul(int8_t** A, int8_t** B, int8_t** C, int rowsA, int colsA, int rowsB, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            C[i][j] = 0.0f;
            for (int k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void gemmini_matmul(int8_t** A, int8_t** B, int8_t** C, int rowsA, int colsA, int rowsB, int colsB) {

    printf("rowsA: %d, colsA: %d, rowsB: %d, colsB: %d\n", rowsA, colsA, rowsB, colsB);

    elem_t full_A[rowsA][colsA];

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            full_A[i][j] = static_cast<elem_t>(A[i][j]);
        }
    }

    std::cout << "Matrix full_A" << std::endl;
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < rowsB; j++) {
            std::cout << static_cast<int>(full_A[i][j]) << " ";
        }
        std::cout << std::endl;
    }

    // std::cout << "Matrix B" << std::endl;
    // for (int i = 0; i < rowsB; i++) {
    //     for (int j = 0; j < colsB; j++) {
    //         std::cout << static_cast<int>(B[i][j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // for(int i = 0; i < rowsB; i++) {
    //     for(int j = 0; j < colsB; j++) {
    //         printf("Address of B[%d][%d]: %p\n", i, j, &(B[i][j]));
    //     }
    // }

    elem_t full_B[rowsB][colsB];

    for (int i = 0; i < rowsB; i++) {
        for (int j = 0; j < colsB; j++) {
            full_B[i][j] = static_cast<elem_t>(B[i][j]);
        }
    }


    std::cout << "Matrix full_B" << std::endl;
    for (int i = 0; i < rowsB; i++) {
        for (int j = 0; j < colsB; j++) {
            std::cout << static_cast<int>(full_B[i][j]) << " ";
        }
        std::cout << std::endl;
    }

    // for(int i = 0; i < rowsB; i++) {
    //     for(int j = 0; j < colsB; j++) {
    //         printf("Address of full_B[%d][%d]: %p\n", i, j, &(full_B[i][j]));
    //     }
    // }

    elem_t full_C[rowsA][colsB];

    tiled_matmul_auto(rowsA, colsB, colsA, 
                    (elem_t*)full_A, (elem_t*)full_B, 
                    NULL, full_C, 
                    colsA, colsB, colsB, colsB, 
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    false, false,
                    false, 0,
                    0,
                    CPU);

    std::cout << "Matrix full_C" << std::endl;
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            std::cout << static_cast<int>(full_C[i][j]) << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            C[i][j] = static_cast<int8_t>(full_C[i][j]);
        }
    }

}

void scatter_cpu(const int n_in, const int n_out, const int c,
                 int8_t** in_feat, int8_t** out_feat, const int* kmap,
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
                int8_t** in_feat, int8_t** out_feat, const int* kmap,
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

void convolution_forward_cpu(int8_t** in_feat, int numRowsIn, int numColsIn,
                             int8_t** out_feat, int numRowsOut, int numColsOut,
                             int8_t*** kernel, int kernelVolume, int kernelRows, int kernelCols,
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
    int8_t** in_buffer = new int8_t*[in_buffer_size];
    for (int i = 0; i < in_buffer_size; i++) {
        in_buffer[i] = new int8_t[numColsIn];
    }

    int8_t** out_buffer = new int8_t*[in_buffer_size];
    for (int i = 0; i < in_buffer_size; i++) {
        out_buffer[i] = new int8_t[numColsOut];
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

void convolution_forward_gemmini(int8_t** in_feat, int numRowsIn, int numColsIn,
                             int8_t** out_feat, int numRowsOut, int numColsOut,
                             int8_t*** kernel, int kernelVolume, int kernelRows, int kernelCols,
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
    int8_t** in_buffer = new int8_t*[in_buffer_size];
    for (int i = 0; i < in_buffer_size; i++) {
        in_buffer[i] = new int8_t[numColsIn];
    }

    int8_t** out_buffer = new int8_t*[in_buffer_size];
    for (int i = 0; i < in_buffer_size; i++) {
        out_buffer[i] = new int8_t[numColsOut];
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