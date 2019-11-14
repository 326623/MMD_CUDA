//
// Created by king-kong on 13/10/19.
//
#include <vector>
#include "clion_helper.h"
#include "mmd_cuda_kernel_impl.h"
#include "utils_cuda.h"

namespace cpptools {
__global__ void TransposeFillKernel(float** distance_matrix, int size) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = thread_id; i < size; i += blockDim.x * gridDim.x) {
    for (int j = i + 1; j < size; ++j) {
      distance_matrix[j][i] = distance_matrix[i][j];
    }
  }
}

// TODO May add ldb
__global__ void TransposeFillKernel(float* distance_matrix, int size) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = thread_id; i < size; i += blockDim.x * gridDim.x) {
    for (int j = i + 1; j < size; ++j) {
      distance_matrix[j * size + i] = distance_matrix[i * size + j];
    }
  }
}

void TransposeFill(float** distance_matrix, int size) {
  TransposeFillKernel<<<256, BLOCK_PER_THREADS>>>(distance_matrix, size);
}

void TransposeFill(float* distance_matrix, int size) {
  TransposeFillKernel<<<256, BLOCK_PER_THREADS>>>(distance_matrix, size);
}

namespace experimental {
namespace rbf {
std::vector<std::vector<float>> MMDSelfGPUKernel(
    const std::vector<std::vector<float>>& x_features, float gamma) {
  int size = static_cast<int>(x_features.size());
  std::vector<int> host_x_offsets(size);
  std::vector<int> host_x_sizes(size);
  std::vector<float> host_x_features(size);

  int offset = 0;
  for (int i = 0; i < size; ++i) {
    host_x_offsets[i] = offset;
    int feature_length = x_features[i].size();
    host_x_sizes[i] = feature_length;
    offset += feature_length;
  }

  host_x_features.resize(offset);
  CudaPtr<float> device_x_features(offset);
  CudaPtr<int> device_x_offsets(size);
  CudaPtr<int> device_x_sizes(size);
  CudaPtr<float> device_distance_matrix(size * size);

  for (int i = 0; i < size; ++i) {
    int offset = host_x_offsets[i];
    for (int j = 0; j < x_features[i].size(); ++j) {
      host_x_features[offset + j] = x_features[i][j];
    }
  }

  float base = std::exp(-gamma);

  device_x_features.CopyToGPU(host_x_features.data(), host_x_features.size());
  device_x_offsets.CopyToGPU(host_x_offsets.data(), host_x_offsets.size());
  device_x_sizes.CopyToGPU(host_x_sizes.data(), host_x_sizes.size());

  auto kernel_method = [base] __device__(float x, float y) -> float {
    return __powf(base, (x - y) * (x - y));
  };

  cpptools::MMDSelfKernelCompute<<<256, BLOCK_PER_THREADS>>>(
      device_x_features.rawPtr(), device_x_offsets.rawPtr(),
      device_x_sizes.rawPtr(), size, device_distance_matrix.rawPtr(),
      kernel_method);
  TransposeFill(device_distance_matrix.rawPtr(), size);

  std::vector<std::vector<float>> distance_matrix(
      size, std::vector<float>(size, 0.0f));
  std::vector<float> host_distance_matrix(size * size);
  device_distance_matrix.CopyToCPU(host_distance_matrix.data(),
                                   host_distance_matrix.size());
  for (int i = 0; i < size; ++i) {
    auto start = host_distance_matrix.cbegin() + i * size;
    auto end = host_distance_matrix.cbegin() + (i + 1) * size;
    std::copy(start, end, distance_matrix[i].data());
  }

  return distance_matrix;
}
}  // namespace rbf
}  // namespace experimental
}  // namespace cpptools
