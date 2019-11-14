//
// Created by king-kong on 23/10/19.
//

// #include "mmd_cuda_rbf.h"
#include <algorithm>
#include <tuple>
#include "clion_helper.h"
#include "mmd_cuda.h"
#include "mmd_cuda_kernel_impl.h"
#include "utils_cuda.h"
#include <iostream>

#define EXPONENTIAL_CONSTANT 2.7182818284590452354
namespace cpptools {
namespace rbf {
template <int kFeatureLength>
void __global__ MMDSelfKernelMulFixed(const float* x_features,
                                      const int* x_offsets, const int* x_sizes,
                                      const int size, float* distance_matrix,
                                      float base);

template <int kFeatureLength>
void __global__ MMDKernelMulFixed(const float* x_features, const float* y_features,
                             const int* x_offsets, const int* y_offsets,
                             const int* x_sizes, const int* y_sizes,
                             const int size_x, const int size_y,
                             float* distance_matrix, float base);

void __global__ MMDOuterProdKernelMul(const float* x_features,
                                      const int* x_offsets, const int* x_sizes,
                                      const int size, float* x_outer_prod,
                                      float base, const int k_feature_length);

void __global__ MMDSelfKernelMul(const float* x_features, const int* x_offsets,
                                 const int* x_sizes, const int size,
                                 float* distance_matrix, float base,
                                 const int k_feature_length);



void __global__ MMDKernelMul(const float* x_features, const float* y_features,
                             const int* x_offsets, const int* y_offsets,
                             const int* x_sizes, const int* y_sizes,
                             const int size_x, const int size_y,
                             float* distance_matrix, float base,
                             const int k_feature_length);

std::vector<float> MMDOuterProdGPU(
    const std::vector<std::vector<float>>& x_features, float gamma) {
  GPUFlatVec gpu_vec = ToFlatGPUMem(x_features);
  CudaPtr<float> device_x_features = std::move(std::get<0>(gpu_vec));
  CudaPtr<int> device_x_offsets = std::move(std::get<1>(gpu_vec));
  CudaPtr<int> device_x_sizes = std::move(std::get<2>(gpu_vec));

  int size = x_features.size();
  CudaPtr<float> device_x_outer_prod(size);

  float base = std::exp(-gamma);
  auto rbf_kernel = [base] __device__(float x, float y) -> float {
    return __powf(base, (x - y) * (x - y));
  };

  cpptools::MMDOuterProdKernel<<<256, BLOCK_PER_THREADS>>>(
      device_x_features.rawPtr(), device_x_offsets.rawPtr(),
      device_x_sizes.rawPtr(), size, device_x_outer_prod.rawPtr(), rbf_kernel);
  cudaCheckError();

  // Finish computation, copy back
  std::vector<float> x_outer_prod(size);
  device_x_outer_prod.CopyToCPU(x_outer_prod);
  return x_outer_prod;
}

std::vector<std::vector<float>> MMDSelfGPU(
    const std::vector<std::vector<float>>& x_features, float gamma) {
  GPUFlatVec gpu_vec = ToFlatGPUMem(x_features);
  CudaPtr<float> device_x_features = std::move(std::get<0>(gpu_vec));
  CudaPtr<int> device_x_offsets = std::move(std::get<1>(gpu_vec));
  CudaPtr<int> device_x_sizes = std::move(std::get<2>(gpu_vec));
  int size = x_features.size();

  float base = std::exp(-gamma);
  auto rbf_kernel = [base] __device__(float x, float y) -> float {
    return __powf(base, (x - y) * (x - y));
  };

  CudaPtr<float> device_distance_matrix(size * size);

  cpptools::MMDSelfKernelCompute<<<256, BLOCK_PER_THREADS>>>(
      device_x_features.rawPtr(), device_x_offsets.rawPtr(),
      device_x_sizes.rawPtr(), size, device_distance_matrix.rawPtr(),
      rbf_kernel);

  TransposeFill(device_distance_matrix.rawPtr(), size);

  cudaCheckError();

  return To2DVec(std::move(device_distance_matrix), size, size);
}

std::vector<std::vector<float>> MMDGPU(
    const std::vector<std::vector<float>>& x_features,
    const std::vector<std::vector<float>>& y_features, float gamma) {
  GPUFlatVec x_gpu_vec = ToFlatGPUMem(x_features);
  GPUFlatVec y_gpu_vec = ToFlatGPUMem(y_features);
  int size_x = x_features.size();
  int size_y = y_features.size();
  CudaPtr<float> device_x_features = std::move(std::get<0>(x_gpu_vec));
  CudaPtr<int> device_x_offsets = std::move(std::get<1>(x_gpu_vec));
  CudaPtr<int> device_x_sizes = std::move(std::get<2>(x_gpu_vec));
  CudaPtr<float> device_y_features = std::move(std::get<0>(y_gpu_vec));
  CudaPtr<int> device_y_offsets = std::move(std::get<1>(y_gpu_vec));
  CudaPtr<int> device_y_sizes = std::move(std::get<2>(y_gpu_vec));
  CudaPtr<float> device_distance_matrix(size_y * size_x);

  float base = std::exp(-gamma);

  auto rbf_kernel = [base] __device__(float x, float y) -> float {
    return __powf(base, (x - y) * (x - y));
  };

  cpptools::MMDKernelCompute<<<256, BLOCK_PER_THREADS>>>(
      device_x_features.rawPtr(), device_y_features.rawPtr(),
      device_x_offsets.rawPtr(), device_y_offsets.rawPtr(),
      device_x_sizes.rawPtr(), device_y_sizes.rawPtr(), size_x, size_y,
      device_distance_matrix.rawPtr(), rbf_kernel);
  cudaCheckError();

  return To2DVec(std::move(device_distance_matrix), size_y, size_x);
}

std::vector<float> MMDOuterProdGPUMul(
    const std::vector<std::vector<std::vector<float>>>& x_features,
    float gamma) {
  GPUFlatVec gpu_vec_x = ToFlatGPUMem(x_features);
  CudaPtr<float> device_x_features = std::move(std::get<0>(gpu_vec_x));
  CudaPtr<int> device_x_offsets = std::move(std::get<1>(gpu_vec_x));
  CudaPtr<int> device_x_sizes = std::move(std::get<2>(gpu_vec_x));
  int size = x_features.size();
  // TODO May need to check error
  const int kFeatureDim = x_features.front().front().size();
  CudaPtr<float> device_x_outer_prod(size);

  float base = std::exp(-gamma);

  cpptools::rbf::MMDOuterProdKernelMul<<<dim3(32, 32), dim3(8, 8)>>>(
      device_x_features.rawPtr(), device_x_offsets.rawPtr(),
      device_x_sizes.rawPtr(), size, device_x_outer_prod.rawPtr(), base,
      kFeatureDim);

  // Finish computation, copy back
  std::vector<float> x_outer_prod(size);
  device_x_outer_prod.CopyToCPU(x_outer_prod);
  return x_outer_prod;
}

std::vector<std::vector<float>> MMDSelfGPUMul(
    const std::vector<std::vector<std::vector<float>>>& x_features,
    float gamma) {
  GPUFlatVec gpu_vec_x = ToFlatGPUMem(x_features);
  CudaPtr<float> device_x_features = std::move(std::get<0>(gpu_vec_x));
  CudaPtr<int> device_x_offsets = std::move(std::get<1>(gpu_vec_x));
  CudaPtr<int> device_x_sizes = std::move(std::get<2>(gpu_vec_x));
  int size = x_features.size();
  // TODO May need to check error
  const int kFeatureDim = x_features.front().front().size();
  CudaPtr<float> device_distance_matrix(size * size);

  float base = std::exp(-gamma);

  switch (kFeatureDim) {
    case 2 : {
      cpptools::rbf::MMDSelfKernelMulFixed<2><<<dim3(32, 32), dim3(8, 8)>>>(
          device_x_features.rawPtr(), device_x_offsets.rawPtr(),
              device_x_sizes.rawPtr(), size, device_distance_matrix.rawPtr(), base);
      break;
    }
    case 3 : {
      cpptools::rbf::MMDSelfKernelMulFixed<3><<<dim3(32, 32), dim3(8, 8)>>>(
          device_x_features.rawPtr(), device_x_offsets.rawPtr(),
              device_x_sizes.rawPtr(), size, device_distance_matrix.rawPtr(), base);
      break;
    }
    case 20: {
      cpptools::rbf::MMDSelfKernelMulFixed<20><<<dim3(32, 32), dim3(8, 8)>>>(
          device_x_features.rawPtr(), device_x_offsets.rawPtr(),
              device_x_sizes.rawPtr(), size, device_distance_matrix.rawPtr(), base);
      break;
    }
    case 30: {
      cpptools::rbf::MMDSelfKernelMulFixed<30><<<dim3(32, 32), dim3(8, 8)>>>(
          device_x_features.rawPtr(), device_x_offsets.rawPtr(),
              device_x_sizes.rawPtr(), size, device_distance_matrix.rawPtr(), base);
      break;
    }
    case 60: {
      cpptools::rbf::MMDSelfKernelMulFixed<60><<<dim3(32, 32), dim3(8, 8)>>>(
          device_x_features.rawPtr(), device_x_offsets.rawPtr(),
              device_x_sizes.rawPtr(), size, device_distance_matrix.rawPtr(), base);
      break;
    }
    default: {
      std::cerr << "Running here means that you need to recompile.\n";
      cpptools::rbf::MMDSelfKernelMul<<<dim3(32, 32), dim3(8, 8)>>>(
          device_x_features.rawPtr(), device_x_offsets.rawPtr(),
              device_x_sizes.rawPtr(), size, device_distance_matrix.rawPtr(), base,
              kFeatureDim);
    }
  }
  TransposeFill(device_distance_matrix.rawPtr(), size);

  return To2DVec(std::move(device_distance_matrix), size, size);
}

std::vector<std::vector<float>> MMDGPUMul(
    const std::vector<std::vector<std::vector<float>>>& x_features,
    const std::vector<std::vector<std::vector<float>>>& y_features,
    float gamma) {
  GPUFlatVec gpu_vec_x = ToFlatGPUMem(x_features);
  CudaPtr<float> device_x_features = std::move(std::get<0>(gpu_vec_x));
  CudaPtr<int> device_x_offsets = std::move(std::get<1>(gpu_vec_x));
  CudaPtr<int> device_x_sizes = std::move(std::get<2>(gpu_vec_x));
  int size_x = x_features.size();

  GPUFlatVec gpu_vec_y = ToFlatGPUMem(y_features);
  CudaPtr<float> device_y_features = std::move(std::get<0>(gpu_vec_y));
  CudaPtr<int> device_y_offsets = std::move(std::get<1>(gpu_vec_y));
  CudaPtr<int> device_y_sizes = std::move(std::get<2>(gpu_vec_y));
  int size_y = y_features.size();

  // TODO May need to check error
  const int kFeatureDim = x_features.front().front().size();
  CudaPtr<float> device_distance_matrix(size_y * size_x);

  float base = std::exp(-gamma);

  switch (kFeatureDim) {
    case 3: {
      cpptools::rbf::MMDKernelMulFixed<3><<<dim3(32, 32), dim3(8, 8)>>>(
          device_x_features.rawPtr(), device_y_features.rawPtr(),
              device_x_offsets.rawPtr(), device_y_offsets.rawPtr(),
              device_x_sizes.rawPtr(), device_y_sizes.rawPtr(), size_x, size_y,
              device_distance_matrix.rawPtr(), base);
      break;
    }
    case 30: {
      cpptools::rbf::MMDKernelMulFixed<30><<<dim3(32, 32), dim3(8, 8)>>>(
          device_x_features.rawPtr(), device_y_features.rawPtr(),
              device_x_offsets.rawPtr(), device_y_offsets.rawPtr(),
              device_x_sizes.rawPtr(), device_y_sizes.rawPtr(), size_x, size_y,
              device_distance_matrix.rawPtr(), base);
      break;
    }
    case 60: {
      cpptools::rbf::MMDKernelMulFixed<60><<<dim3(32, 32), dim3(8, 8)>>>(
          device_x_features.rawPtr(), device_y_features.rawPtr(),
              device_x_offsets.rawPtr(), device_y_offsets.rawPtr(),
              device_x_sizes.rawPtr(), device_y_sizes.rawPtr(), size_x, size_y,
              device_distance_matrix.rawPtr(), base);
      break;
    }
    default: {
      std::cerr << "Running here means that you need to recompile.\n";
      cpptools::rbf::MMDKernelMul<<<dim3(32, 32), dim3(8, 8)>>>(
          device_x_features.rawPtr(), device_y_features.rawPtr(),
              device_x_offsets.rawPtr(), device_y_offsets.rawPtr(),
              device_x_sizes.rawPtr(), device_y_sizes.rawPtr(), size_x, size_y,
              device_distance_matrix.rawPtr(), base, kFeatureDim);
    }
  }

  return To2DVec(std::move(device_distance_matrix), size_y, size_x);
}

//void __global__ MMDSelfKernelCompute(float* x_features, int* x_offsets,
//                                     int* x_sizes, int size,
//                                     float* distance_matrix, float base) {
//  // Assume passing in kFeatureLength of blockDim.y
//  int thread_id_x = threadIdx.x;
//  int thread_id_y = threadIdx.y;
//  int block_id = blockIdx.x;
//
//  for (int index_i = 0; index_i < size; ++index_i) {
//    for (int index_j = index_i + block_id; index_j < size;
//         index_j += gridDim.x) {
//      float sum_kernel = 0.0f;
//      float* features_xi = x_features + x_offsets[index_i];
//      float* features_xj = x_features + x_offsets[index_j];
//      int len_xi = x_sizes[index_i];
//      int len_xj = x_sizes[index_j];
//      for (int i = thread_id_x; i < len_xi; i += blockDim.x) {
//        for (int j = thread_id_y; j < len_xj; j += blockDim.y) {
//          float tmp_diff = features_xi[i] - features_xj[j];
//          tmp_diff = tmp_diff * tmp_diff;
//          // performs exp(-gamma |||xi - xj||^2) = pow(exp(-gamma), ||xi -
//          // xj||^2)
//          sum_kernel += __powf(base, tmp_diff);
//        }
//      }
//
//      sum_kernel = block_reduction_sum(sum_kernel);
//      if (thread_id_x == 0 && thread_id_y == 0) {
//        distance_matrix[index_i * size + index_j] =
//            sum_kernel / static_cast<float>(len_xi * len_xj);
//      }
//    }
//  }
//}

template <int kFeatureLength>
void __global__ MMDSelfKernelMulFixed(const float* x_features,
                                      const int* x_offsets, const int* x_sizes,
                                      const int size, float* distance_matrix,
                                      float base) {
  // Assume passing in kFeatureLength of blockDim.y
  int thread_id_x = threadIdx.x;
  int thread_id_y = threadIdx.y;
  int block_id_x = blockIdx.x;
  int block_id_y = blockIdx.y;

  for (int index_i = block_id_x; index_i < size; index_i += gridDim.x) {
    for (int index_j = index_i + block_id_y; index_j < size;
         index_j += gridDim.y) {
      float sum_kernel = 0.0f;
      const float* features_xi = x_features + x_offsets[index_i];
      const float* features_xj = x_features + x_offsets[index_j];
      int len_xi = x_sizes[index_i];
      int len_xj = x_sizes[index_j];

      for (int i = thread_id_x; i < len_xi; i += blockDim.x) {
        const float* features_xi_i = features_xi + i * kFeatureLength;
        for (int j = thread_id_y; j < len_xj; j += blockDim.y) {
          const float* features_xj_j = features_xj + j * kFeatureLength;
          // performs ||xi - xj||^2
          float sub_sum = 0.0f;
#pragma unroll
          for (int k = 0; k < kFeatureLength; ++k) {
            float tmp_diff = features_xi_i[k] - features_xj_j[k];
            tmp_diff = tmp_diff * tmp_diff;
            sub_sum += tmp_diff;
          }
          sum_kernel += __powf(base, sub_sum);
        }
      }

      sum_kernel = block_reduction_sum(sum_kernel);
      if (thread_id_x == 0 && thread_id_y == 0) {
        distance_matrix[index_i * size + index_j] =
            (sum_kernel / static_cast<float>(len_xi * len_xj));
      }
    }
  }
}

void __global__ MMDOuterProdKernelMul(const float* x_features,
                                      const int* x_offsets, const int* x_sizes,
                                      const int size, float* x_outer_prod,
                                      float base, const int k_feature_length) {
  // Assume passing in k_feature_length of blockDim.y
  int thread_id_x = threadIdx.x;
  int thread_id_y = threadIdx.y;
  int block_id_x = blockIdx.x;

  for (int index_i = block_id_x; index_i < size; index_i += gridDim.x) {
    float sum_kernel = 0.0f;
    const float* features_xi = x_features + x_offsets[index_i];
    int len_xi = x_sizes[index_i];

    for (int i = thread_id_x; i < len_xi; i += blockDim.x) {
      const float* features_xi_i = features_xi + i * k_feature_length;
      for (int j = thread_id_y; j < len_xi; j += blockDim.y) {
        const float* features_xi_j = features_xi + j * k_feature_length;
        // performs ||xi - xj||^2
        float sub_sum = 0.0f;
        for (int k = 0; k < k_feature_length; ++k) {
          float tmp_diff = features_xi_i[k] - features_xi_j[k];
          tmp_diff = tmp_diff * tmp_diff;
          sub_sum += tmp_diff;
        }
        sum_kernel += __powf(base, sub_sum);
      }
    }

    sum_kernel = block_reduction_sum(sum_kernel);
    if (thread_id_x == 0 && thread_id_y == 0) {
      x_outer_prod[index_i] =
          (sum_kernel / static_cast<float>(len_xi * len_xi));
    }
  }
}

void __global__ MMDSelfKernelMul(const float* x_features, const int* x_offsets,
                                 const int* x_sizes, const int size,
                                 float* distance_matrix, float base,
                                 const int k_feature_length) {
  // Assume passing in k_feature_length of blockDim.y
  int thread_id_x = threadIdx.x;
  int thread_id_y = threadIdx.y;
  int block_id_x = blockIdx.x;
  int block_id_y = blockIdx.y;

  for (int index_i = block_id_x; index_i < size; index_i += gridDim.x) {
    for (int index_j = index_i + block_id_y; index_j < size;
         index_j += gridDim.y) {
      float sum_kernel = 0.0f;
      const float* features_xi = x_features + x_offsets[index_i];
      const float* features_xj = x_features + x_offsets[index_j];
      int len_xi = x_sizes[index_i];
      int len_xj = x_sizes[index_j];

      for (int i = thread_id_x; i < len_xi; i += blockDim.x) {
        const float* features_xi_i = features_xi + i * k_feature_length;
        for (int j = thread_id_y; j < len_xj; j += blockDim.y) {
          const float* features_xj_j = features_xj + j * k_feature_length;
          // performs ||xi - xj||^2
          float sub_sum = 0.0f;
          for (int k = 0; k < k_feature_length; ++k) {
            float tmp_diff = features_xi_i[k] - features_xj_j[k];
            tmp_diff = tmp_diff * tmp_diff;
            sub_sum += tmp_diff;
          }
          sum_kernel += __powf(base, sub_sum);
        }
      }

      sum_kernel = block_reduction_sum(sum_kernel);
      if (thread_id_x == 0 && thread_id_y == 0) {
        distance_matrix[index_i * size + index_j] =
            (sum_kernel / static_cast<float>(len_xi * len_xj));
      }
    }
  }
}

template <int kFeatureLength>
void __global__ MMDKernelMulFixed(const float* x_features, const float* y_features,
                                  const int* x_offsets, const int* y_offsets,
                                  const int* x_sizes, const int* y_sizes,
                                  const int size_x, const int size_y,
                                  float* distance_matrix, float base) {
  // Assume passing in k_feature_length of blockDim.y
  int thread_id_x = threadIdx.x;
  int thread_id_y = threadIdx.y;
  int block_id_x = blockIdx.x;
  int block_id_y = blockIdx.y;

  for (int index_i = block_id_x; index_i < size_x; index_i += gridDim.x) {
    for (int index_j = block_id_y; index_j < size_y; index_j += gridDim.y) {
      float sum_kernel = 0.0f;
      const float* features_xi = x_features + x_offsets[index_i];
      const float* features_xj = y_features + y_offsets[index_j];
      int len_xi = x_sizes[index_i];
      int len_xj = y_sizes[index_j];

      for (int i = thread_id_x; i < len_xi; i += blockDim.x) {
        const float* features_xi_i = features_xi + i * kFeatureLength;
        for (int j = thread_id_y; j < len_xj; j += blockDim.y) {
          const float* features_xj_j = features_xj + j * kFeatureLength;
          // performs ||xi - xj||^2
          float sub_sum = 0.0f;
          for (int k = 0; k < kFeatureLength; ++k) {
            float tmp_diff = features_xi_i[k] - features_xj_j[k];
            tmp_diff = tmp_diff * tmp_diff;
            sub_sum += tmp_diff;
          }
          sum_kernel += __powf(base, sub_sum);
        }
      }

      sum_kernel = block_reduction_sum(sum_kernel);
      if (thread_id_x == 0 && thread_id_y == 0) {
        distance_matrix[index_i + index_j * size_x] =
            (sum_kernel / static_cast<float>(len_xi * len_xj));
      }
    }
  }
}

// TODO: add test case
void __global__ MMDKernelMul(const float* x_features, const float* y_features,
                             const int* x_offsets, const int* y_offsets,
                             const int* x_sizes, const int* y_sizes,
                             const int size_x, const int size_y,
                             float* distance_matrix, float base,
                             const int k_feature_length) {
  // Assume passing in k_feature_length of blockDim.y
  int thread_id_x = threadIdx.x;
  int thread_id_y = threadIdx.y;
  int block_id_x = blockIdx.x;
  int block_id_y = blockIdx.y;

  for (int index_i = block_id_x; index_i < size_x; index_i += gridDim.x) {
    for (int index_j = block_id_y; index_j < size_y; index_j += gridDim.y) {
      float sum_kernel = 0.0f;
      const float* features_xi = x_features + x_offsets[index_i];
      const float* features_xj = y_features + y_offsets[index_j];
      int len_xi = x_sizes[index_i];
      int len_xj = y_sizes[index_j];

      for (int i = thread_id_x; i < len_xi; i += blockDim.x) {
        const float* features_xi_i = features_xi + i * k_feature_length;
        for (int j = thread_id_y; j < len_xj; j += blockDim.y) {
          const float* features_xj_j = features_xj + j * k_feature_length;
          // performs ||xi - xj||^2
          float sub_sum = 0.0f;
          for (int k = 0; k < k_feature_length; ++k) {
            float tmp_diff = features_xi_i[k] - features_xj_j[k];
            tmp_diff = tmp_diff * tmp_diff;
            sub_sum += tmp_diff;
          }
          sum_kernel += __powf(base, sub_sum);
        }
      }

      sum_kernel = block_reduction_sum(sum_kernel);
      if (thread_id_x == 0 && thread_id_y == 0) {
        distance_matrix[index_i + index_j * size_x] =
            (sum_kernel / static_cast<float>(len_xi * len_xj));
      }
    }
  }
}
}  // namespace rbf
}  // namespace cpptools
