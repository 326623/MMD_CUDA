//
// Created by king-kong on 23/10/19.
//
#include <cuda_runtime.h>
#include <cmath>
#include <tuple>
#include "clion_helper.h"
#include "mmd_cuda.h"
#include "mmd_cuda_kernel_impl.h"
#include "utils_cuda.h"

namespace cpptools {
namespace linear {
void MMDGPUKernelOuterProdImpl(float* x_features, int* x_offsets, int* x_sizes,
                               int size, float* x_outer_prod);

void MMDSelfGPUKernelImpl(float* x_features, int* x_offsets, int* x_sizes,
                          int size, float* distance_matrix);

void MMDGPUKernelImpl(float* x_features, float* y_features, int* x_offsets,
                      int* y_offsets, int* x_sizes, int* y_sizes, int size_x,
                      int size_y, float* distance_matrix);

__global__ void FeatureWiseSum(float* x_features, int* x_offsets, int* x_sizes,
                               int size_x, float* x_average);

__global__ void MMDOuterProdComputeWithSum(float* x_average, int size_x,
                                           float* x_outer_prod);

__global__ void MMDSelfComputeWithSum(float* x_average, int size_x,
                                      float* distance_matrix);

__global__ void MMDComputeWithSum(float* x_average, float* y_average,
                                  int size_x, int size_y,
                                  float* distance_matrix);

std::vector<float> MMDOuterProdGPU(
    const std::vector<std::vector<float>>& x_features) {
  GPUFlatVec x_gpu_vec = ToFlatGPUMem(x_features);
  CudaPtr<float> device_x_features = std::move(std::get<0>(x_gpu_vec));
  CudaPtr<int> device_offsets = std::move(std::get<1>(x_gpu_vec));
  CudaPtr<int> device_sizes = std::move(std::get<2>(x_gpu_vec));
  int size = x_features.size();

  CudaPtr<float> device_x_outer_prod(size);

  cpptools::linear::MMDGPUKernelOuterProdImpl(
      device_x_features.rawPtr(), device_offsets.rawPtr(),
      device_sizes.rawPtr(), size, device_x_outer_prod.rawPtr());
  cudaCheckError();

  // Finish computation, copy back
  std::vector<float> x_outer_prod(x_features.size());
  device_x_outer_prod.CopyToCPU(x_outer_prod);
  return x_outer_prod;
}

std::vector<std::vector<float>> MMDSelfGPU(
    const std::vector<std::vector<float>>& x_features) {
  // CudaMMDSelfArgs args = PrepareMMDSelfArgs(x_features);
  GPUFlatVec x_gpu_vec = ToFlatGPUMem(x_features);
  CudaPtr<float> device_x_features = std::move(std::get<0>(x_gpu_vec));
  CudaPtr<int> device_offsets = std::move(std::get<1>(x_gpu_vec));
  CudaPtr<int> device_sizes = std::move(std::get<2>(x_gpu_vec));
  int size = x_features.size();

  CudaPtr<float> device_distance_matrix(size * size);

  cpptools::linear::MMDSelfGPUKernelImpl(
      device_x_features.rawPtr(), device_offsets.rawPtr(),
      device_sizes.rawPtr(), size, device_distance_matrix.rawPtr());
  cpptools::TransposeFill(device_distance_matrix.rawPtr(), size);
  cudaCheckError();

  // Finish computation, copy back
  return To2DVec(std::move(device_distance_matrix), size, size);
}

std::vector<std::vector<float>> MMDGPU(
    const std::vector<std::vector<float>>& x_features,
    const std::vector<std::vector<float>>& y_features) {
  GPUFlatVec x_gpu_vec = ToFlatGPUMem(x_features);
  GPUFlatVec y_gpu_vec = ToFlatGPUMem(y_features);

  CudaPtr<float> device_x_features = std::move(std::get<0>(x_gpu_vec));
  CudaPtr<int> device_x_offsets = std::move(std::get<1>(x_gpu_vec));
  CudaPtr<int> device_x_sizes = std::move(std::get<2>(x_gpu_vec));

  CudaPtr<float> device_y_features = std::move(std::get<0>(y_gpu_vec));
  CudaPtr<int> device_y_offsets = std::move(std::get<1>(y_gpu_vec));
  CudaPtr<int> device_y_sizes = std::move(std::get<2>(y_gpu_vec));

  int size_x = x_features.size();
  int size_y = y_features.size();

  CudaPtr<float> device_distance_matrix(size_y * size_x);

  cpptools::linear::MMDGPUKernelImpl(
      device_x_features.rawPtr(), device_y_features.rawPtr(),
      device_x_offsets.rawPtr(), device_y_offsets.rawPtr(),
      device_x_sizes.rawPtr(), device_y_sizes.rawPtr(), size_x, size_y,
      device_distance_matrix.rawPtr());

  cudaCheckError();

  return To2DVec(std::move(device_distance_matrix), size_y, size_x);
}

void MMDGPUKernelOuterProdImpl(float* x_features, int* x_offsets, int* x_sizes,
                               int size, float* x_outer_prod) {
  CudaPtr<float> x_average(size);
  cpptools::linear::FeatureWiseSum<<<256, BLOCK_PER_THREADS>>>(
      x_features, x_offsets, x_sizes, size, x_average.rawPtr());
  cpptools::linear::MMDOuterProdComputeWithSum<<<256, BLOCK_PER_THREADS>>>(
      x_average.rawPtr(), size, x_outer_prod);
}

void MMDSelfGPUKernelImpl(float* x_features, int* x_offsets, int* x_sizes,
                          int size, float* distance_matrix) {
  CudaPtr<float> x_average(size);
  cpptools::linear::FeatureWiseSum<<<256, BLOCK_PER_THREADS>>>(
      x_features, x_offsets, x_sizes, size, x_average.rawPtr());
  cpptools::linear::MMDSelfComputeWithSum<<<256, BLOCK_PER_THREADS>>>(
      x_average.rawPtr(), size, distance_matrix);
}

void MMDGPUKernelImpl(float* x_features, float* y_features, int* x_offsets,
                      int* y_offsets, int* x_sizes, int* y_sizes, int size_x,
                      int size_y, float* distance_matrix) {
  CudaPtr<float> x_average(size_x);
  CudaPtr<float> y_average(size_y);
  cpptools::linear::FeatureWiseSum<<<256, BLOCK_PER_THREADS>>>(
      x_features, x_offsets, x_sizes, size_x, x_average.rawPtr());
  cpptools::linear::FeatureWiseSum<<<256, BLOCK_PER_THREADS>>>(
      y_features, y_offsets, y_sizes, size_y, y_average.rawPtr());
  cpptools::linear::MMDComputeWithSum<<<256, BLOCK_PER_THREADS>>>(
      x_average.rawPtr(), y_average.rawPtr(), size_x, size_y, distance_matrix);
}

__global__ void FeatureWiseSum(float* x_features, int* x_offsets, int* x_sizes,
                               int size_x, float* x_average) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;

  for (int i = block_id; i < size_x; i += gridDim.x) {
    int length_x_feature = x_sizes[i];
    float* x_features_i = x_features + x_offsets[i];
    float thread_accumulator = 0.0f;

    for (int j = thread_id; j < length_x_feature; j += blockDim.x) {
      thread_accumulator += x_features_i[j];
    }

    thread_accumulator = block_reduction_sum(thread_accumulator);
    if (thread_id == 0) {
      x_average[i] = thread_accumulator / static_cast<float>(length_x_feature);
    }
  }
}

__global__ void MMDOuterProdComputeWithSum(float* x_average, int size_x,
                                           float* x_outer_prod) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  for (int i = block_id * blockDim.x + thread_id; i < size_x;
       i += gridDim.x * blockDim.x) {
    x_outer_prod[i] = x_average[i] * x_average[i];
  }
}

__global__ void MMDSelfComputeWithSum(float* x_average, int size_x,
                                      float* distance_matrix) {
  // int thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  for (int i = block_id; i < size_x; i += gridDim.x) {
    for (int j = thread_id + i; j < size_x; j += blockDim.x) {
      distance_matrix[i * size_x + j] = x_average[i] * x_average[j];
    }
  }
}

__global__ void MMDComputeWithSum(float* x_average, float* y_average,
                                  int size_x, int size_y,
                                  float* distance_matrix) {
  // int thread_global_id = blockIdx.x * blockDim.x + threadIdx.x;
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  for (int i = block_id; i < size_y; i += gridDim.x) {
    for (int j = thread_id; j < size_x; j += blockDim.x) {
      distance_matrix[i * size_x + j] = y_average[i] * x_average[j];
    }
  }
}
}  // namespace linear
}  // namespace cpptools
