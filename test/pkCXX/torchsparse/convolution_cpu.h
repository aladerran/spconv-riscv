#pragma once
#include <vector>
#include <algorithm>
#include <stdexcept>

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

void mm_out(float *out_data, const float *in_data, const float *kernel_data, 
            int nrows, int k, int ncols) {
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      out_data[i * ncols + j] = 0;
      for (int l = 0; l < k; l++) {
        out_data[i * ncols + j] += in_data[i * k + l] * kernel_data[l * ncols + j];
      }
    }
  }
}

void convolution_forward_cpu(const std::vector<float>& in_feat,
                             std::vector<float>& out_feat,
                             const std::vector<float>& kernel,
                             const std::vector<int>& neighbor_map,
                             const std::vector<int>& neighbor_offset, 
                             const bool transpose, int c, int kernel_volume) {
  
  if (in_feat.size() / c != kernel.size() / (c * kernel_volume)) {
    throw std::invalid_argument("Input feature size and kernel size mismatch");
  }

  int out_nrows = out_feat.size() / c;
  out_feat.resize(out_nrows * kernel_volume);
  std::fill(out_feat.begin(), out_feat.end(), 0.0f);

  int in_buffer_size = 1;
  bool flag = false;
  // memory optimization
  if (kernel_volume % 2 && out_nrows == in_feat.size() / c) {
    flag = true;
    in_buffer_size =
        *std::max_element(neighbor_offset.begin(),
                          neighbor_offset.begin() + kernel_volume / 2);
    in_buffer_size =
        std::max(in_buffer_size,
                 *std::max_element(
                     neighbor_offset.begin() + kernel_volume / 2 + 1,
                     neighbor_offset.end()));
    in_buffer_size = std::max(in_buffer_size, 1);

    mm_out(&out_feat[0], &in_feat[0], &kernel[kernel_volume / 2 * c * c], 
           out_nrows, c, c);
  } else {
    in_buffer_size =
        *std::max_element(neighbor_offset.begin(), neighbor_offset.end());
  }

  std::vector<float> in_buffer(in_buffer_size * c, 0.0f);
  std::vector<float> out_buffer(in_buffer_size * c, 0.0f);
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
    gather_cpu(neighbor_offset[i], in_feat.size() / c, c,
               &in_feat[0], &in_buffer[0],
               &neighbor_map[cur_offset], transpose);

    // matmul
    mm_out(&out_buffer[0], &in_buffer[0], &kernel[i * c * c], 
           neighbor_offset[i], c, c);

    // scatter
    scatter_cpu(neighbor_offset[i], out_nrows, c,
                &out_buffer[0], &out_feat[0], 
                &neighbor_map[cur_offset], transpose);
    cur_offset += 2 * neighbor_offset[i];
  }
}
