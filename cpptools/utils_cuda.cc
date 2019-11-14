//
// Created by king-kong on 23/10/19.
//
#include "utils_cuda.h"
#include <vector>

std::vector<std::vector<float>> To2DVec(CudaPtr<float> device_distance_matrix,
                                        int row_size, int col_size) {
  std::vector<float> host_distance_matrix(row_size * col_size);
  device_distance_matrix.CopyToCPU(host_distance_matrix);
  std::vector<std::vector<float>> res_vec(row_size,
                                          std::vector<float>(col_size));

  for (int i = 0; i < row_size; ++i) {
    auto start = host_distance_matrix.cbegin() + i * col_size;
    auto end = host_distance_matrix.cbegin() + (i + 1) * col_size;
    std::copy(start, end, res_vec[i].data());
  }

  return res_vec;
}

GPUFlatVec ToFlatGPUMem(const std::vector<std::vector<float>>& features) {
  int size = features.size();
  std::vector<int> offsets(size);
  std::vector<int> sizes(size);
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    offsets[i] = offset;
    int length = features[i].size();
    sizes[i] = length;
    offset += length;
  }

  std::vector<float> features_flat(offset);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < sizes[i]; ++j) {
      features_flat[j + offsets[i]] = features[i][j];
    }
  }

  CudaPtr<float> device_x_features(features_flat.size());
  CudaPtr<int> device_x_offsets(offsets.size());
  CudaPtr<int> device_x_sizes(sizes.size());

  device_x_features.CopyToGPU(features_flat);
  device_x_offsets.CopyToGPU(offsets);
  device_x_sizes.CopyToGPU(sizes);

  return GPUFlatVec(std::move(device_x_features), std::move(device_x_offsets),
                    std::move(device_x_sizes));
}

GPUFlatVec ToFlatGPUMem(
    const std::vector<std::vector<std::vector<float>>>& x_features) {
  // TODO: maybe add some error checking.
  const int kFeatureDim = x_features.front().front().size();
  int size = static_cast<int>(x_features.size());
  std::vector<int> host_x_offsets(size);
  std::vector<int> host_x_sizes(size);

  int offset = 0;
  for (int i = 0; i < size; ++i) {
    host_x_offsets[i] = offset;
    host_x_sizes[i] = x_features[i].size();
    offset += x_features[i].size() * kFeatureDim;
  }

  std::vector<float> host_x_features(offset);

  CudaPtr<float> device_x_features(offset);
  CudaPtr<int> device_x_offsets(size);
  CudaPtr<int> device_x_sizes(size);

  for (int i = 0; i < size; ++i) {
    int offset = host_x_offsets[i];
    for (int k = 0; k < kFeatureDim; ++k) {
      for (int j = 0; j < x_features[i].size(); ++j) {
        // host_x_features[offset + k * x_features[i].size() + j] =
        // x_features[i][j][k];
        host_x_features[offset + j * kFeatureDim + k] = x_features[i][j][k];
      }
    }
  }

  device_x_features.CopyToGPU(host_x_features);
  device_x_offsets.CopyToGPU(host_x_offsets);
  device_x_sizes.CopyToGPU(host_x_sizes);

  return GPUFlatVec(std::move(device_x_features),
                    std::move(device_x_offsets),
                    std::move(device_x_sizes));
}