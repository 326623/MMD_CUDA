//
// Created by king-kong on 22/10/19.
//

#ifndef HPCORPIET_CPPTOOLS_MMD_CUDA_KERNEL_IMPL_H_
#define HPCORPIET_CPPTOOLS_MMD_CUDA_KERNEL_IMPL_H_

#include <stdlib.h>
#include <cmath>
#include "clion_helper.h"
#define BLOCK_PER_THREADS 256
// MTile Larger can reduce mem reads
#define MTILE_SIZE BLOCK_PER_THREADS * 4
#define NTILE_SIZE BLOCK_PER_THREADS * 4
#define THREADS_WORKING_SIZE 32
#define WARP_SIZE 32

namespace cpptools {
// Since the use of intrinsics are very close to HW. I cannot utilize any 2D dim
// here without first converting it to 1D. So, don't use it for 2D or 3D.
// Do note that calling this function has the potential to add up garbage value,
// which ends up returning the wrong sum. So please ensure the thread that is
// not active in the summing having zero as initial value.
__inline__ __device__ float WarpReduceSum(float value) {
  // Use XOR mode to perform butterfly reduction
#pragma unroll
  for (int i = 16; i >= 1; i /= 2)
    value += __shfl_xor_sync(0xffffffff, value, i, 32);
  return value;
}

int __inline__ __device__ roundUp(int n, int div) { return (n / div) * div; }

// Underlying assume: Cuda maximum thread per block is 1024, 32 * 32. two full
// reduction at max TODO: May need to add some constraints here
float __inline__ __device__ block_reduction_sum(float value) {
  __shared__ float shared[WARP_SIZE];
  int thread_id = threadIdx.x + threadIdx.y * blockDim.x +
                  threadIdx.z * blockDim.x * blockDim.y;
  int block_size = blockDim.x * blockDim.y * blockDim.z;
  int lane = thread_id % WARP_SIZE;
  int wid = thread_id / WARP_SIZE;
  value = WarpReduceSum(value);
  if (block_size <= WARP_SIZE) {
    return value;
  }
  if (lane == 0) shared[wid] = value;
  __syncthreads();
  value = (thread_id < block_size / WARP_SIZE) ? shared[lane] : 0;
  if (wid == 0) value = WarpReduceSum(value);
  return value;
}

// assume blockDim.x == 2^n
void __inline__ __device__ x_block_reduction_sum(float* array, float* value) {
  int xyz_offset =
      threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  array[xyz_offset + threadIdx.x] = *value;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      array[xyz_offset + threadIdx.x] +=
          array[xyz_offset + threadIdx.x + offset];
    }
    __syncthreads();
  }

  *value = array[xyz_offset];
  __syncthreads();
}

// assume blockDim.y == 2^n
void __inline__ __device__ y_block_reduction_sum(float* array, float* value) {
  int xyz_offset =
      threadIdx.x * blockDim.y + threadIdx.z * blockDim.y * blockDim.x;
  array[xyz_offset + threadIdx.y] = *value;
  __syncthreads();

  for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
    if (threadIdx.y < offset) {
      array[xyz_offset + threadIdx.y] +=
          array[xyz_offset + threadIdx.y + offset];
    }
    __syncthreads();
  }

  *value = array[xyz_offset];
  __syncthreads();
}

// assume blockDim.z == 2^n
void __inline__ __device__ z_block_reduction_sum(float* array, float* value) {
  int xyz_offset =
      threadIdx.x * blockDim.z + threadIdx.y * blockDim.z * blockDim.x;
  array[xyz_offset + threadIdx.z] = *value;
  __syncthreads();

  for (int offset = blockDim.z / 2; offset > 0; offset /= 2) {
    if (threadIdx.z < offset) {
      array[xyz_offset + threadIdx.z] +=
          array[xyz_offset + threadIdx.z + offset];
    }
    __syncthreads();
  }

  *value = array[xyz_offset];
  __syncthreads();
}

template <int BLOCK_THREADS>
void __inline__ __device__ VectorLoad(float* __restrict__ destination,
                                      const float* source, int size,
                                      int thread_id) {
  for (int i = thread_id; i < size; i += BLOCK_THREADS) {
#if __CUDA_ARCH__ >= 350
    destination[i] = __ldg(source + i);
#else
    destination[i] = source[i];
#endif
  }
}

void __inline__ __device__ VectorLoad(float* __restrict__ destination,
                                      const float* source, int size,
                                      int thread_id, int block_size) {
  for (int i = thread_id; i < size; i += block_size) {
#if __CUDA_ARCH__ >= 350
    destination[i] = __ldg(source + i);
#else
    destination[i] = source[i];
  }
#endif
  }

  template <typename BinaryOp>
  __global__ void MMDOuterProdKernel(
      float* x_features, int* x_offsets, int* x_sizes, int size,
      float* x_outer_prod, BinaryOp kernel_method) {
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;

    for (int xi = block_id; xi < size; xi += gridDim.x) {
      float* x_features_xi = x_features + x_offsets[xi];
      int len_xi = x_sizes[xi];

      float kernel_sum = 0.0f;
      for (int i = thread_id; i < len_xi; i += blockDim.x) {
        for (int j = 0; j < len_xi; ++j) {
          kernel_sum += kernel_method(x_features_xi[i], x_features_xi[j]);
        }
      }

      kernel_sum = block_reduction_sum(kernel_sum);

      if (thread_id == 0) {
        x_outer_prod[xi] = kernel_sum / static_cast<float>(len_xi * len_xi);
      }
    }
  }

  template <typename BinaryOp>
  void __global__ MMDSelfKernelCompute(
      float* x_features, int* x_offsets, int* x_sizes, int size,
      float* distance_matrix, BinaryOp kernel_method) {
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    // __shared__ float sub_sum[BLOCK_PER_THREADS];
    __shared__ float frag_x[MTILE_SIZE];
    __shared__ float frag_y[NTILE_SIZE];
    float local_frag_y[THREADS_WORKING_SIZE];

    // Total work: (size - 1 + size - 2 + ... + 1) == (size - 1) * size / 2

    // const auto& x_features = *x_features_ptr;
    for (int xi = 0; xi < size; ++xi) {
      // divice workloads for every block
      int num_workloads = size - xi;
      int workloads_per_block = (num_workloads + gridDim.x - 1) / gridDim.x;
      int start = workloads_per_block * block_id + xi;
      int end = min(start + workloads_per_block, size);

      for (int xj = start; xj < end; ++xj) {
        int len_xi = x_sizes[xi];
        int len_xj = x_sizes[xj];
        float* features_xi = x_features + x_offsets[xi];
        float* features_xj = x_features + x_offsets[xj];

        float sub_kernel_sum = 0.0f;

        int decimal_x = roundUp(len_xi, MTILE_SIZE);
        int decimal_y = roundUp(len_xj, NTILE_SIZE);
        // decimal_x && decimal_y
        for (int x_frag_start = 0; x_frag_start < decimal_x;
             x_frag_start += MTILE_SIZE) {
          // load frag_x
          VectorLoad<BLOCK_PER_THREADS>(frag_x, features_xi + x_frag_start,
                                        MTILE_SIZE, thread_id);
          for (int y_frag_start = 0; y_frag_start < decimal_y;
               y_frag_start += NTILE_SIZE) {
            // Load frag_y
            VectorLoad<BLOCK_PER_THREADS>(frag_y, features_xj + y_frag_start,
                                          NTILE_SIZE, thread_id);
            __syncthreads();

#pragma unroll
            for (int j = 0; j < NTILE_SIZE; j += THREADS_WORKING_SIZE) {
#pragma unroll
              for (int sub_j = 0; sub_j < THREADS_WORKING_SIZE; ++sub_j) {
                local_frag_y[sub_j] = frag_y[j + sub_j];
              }
              for (int i = thread_id; i < MTILE_SIZE; i += blockDim.x) {
                for (int sub_j = 0; sub_j < THREADS_WORKING_SIZE; ++sub_j) {
                  sub_kernel_sum +=
                      kernel_method(frag_x[i], local_frag_y[sub_j]);
                }
              }
            }
            __syncthreads();
          }
        }

        // decimal_x && tail_y
        {
          int frag_y_size = len_xj - decimal_y;
          VectorLoad<BLOCK_PER_THREADS>(frag_y, features_xj + decimal_y,
                                        frag_y_size, thread_id);
          // Tail loop of THREADS_WORKING_SIZE: to enable local_cache without if
          // branch
          int decimal_tail_y = roundUp(frag_y_size, THREADS_WORKING_SIZE);
          for (int x_frag_start = 0; x_frag_start < decimal_x;
               x_frag_start += MTILE_SIZE) {
            // load frag_x
            VectorLoad<BLOCK_PER_THREADS>(frag_x, features_xi + x_frag_start,
                                          MTILE_SIZE, thread_id);

            __syncthreads();

#pragma unroll
            for (int j = 0; j < decimal_tail_y; j += THREADS_WORKING_SIZE) {
#pragma unroll
              for (int sub_j = 0; sub_j < THREADS_WORKING_SIZE; ++sub_j) {
                local_frag_y[sub_j] = frag_y[j + sub_j];
              }
              for (int i = thread_id; i < MTILE_SIZE; i += blockDim.x) {
                for (int sub_j = 0; sub_j < THREADS_WORKING_SIZE; ++sub_j) {
                  sub_kernel_sum +=
                      kernel_method(frag_x[i], local_frag_y[sub_j]);
                }
              }
            }

            // Tail loop of THREADS_WORKING_SIZE
            for (int i = thread_id; i < MTILE_SIZE; i += blockDim.x) {
              for (int j = decimal_tail_y; j < frag_y_size; ++j) {
                sub_kernel_sum += kernel_method(frag_x[i], frag_y[j]);
              }
            }
            __syncthreads();
          }
        }

        // decimal_y && tail_x
        {
          int x_frag_size = len_xi - decimal_x;
          VectorLoad<BLOCK_PER_THREADS>(frag_x, features_xi + decimal_x,
                                        x_frag_size, thread_id);
          for (int y_frag_start = 0; y_frag_start < decimal_y;
               y_frag_start += NTILE_SIZE) {
            // Load frag_y
            VectorLoad<BLOCK_PER_THREADS>(frag_y, features_xj + y_frag_start,
                                          NTILE_SIZE, thread_id);
            __syncthreads();

#pragma unroll
            for (int j = 0; j < NTILE_SIZE; j += THREADS_WORKING_SIZE) {
#pragma unroll
              for (int sub_j = 0; sub_j < THREADS_WORKING_SIZE; ++sub_j) {
                local_frag_y[sub_j] = frag_y[j + sub_j];
              }
              for (int i = thread_id; i < x_frag_size; i += blockDim.x) {
                for (int sub_j = 0; sub_j < THREADS_WORKING_SIZE; ++sub_j) {
                  sub_kernel_sum +=
                      kernel_method(frag_x[i], local_frag_y[sub_j]);
                }
              }
            }
            __syncthreads();
          }
        }

        // tail_x && tail_y
        {
          // Tail of loop
          int x_frag_size = len_xi - decimal_x;
          int frag_y_size = len_xj - decimal_y;
          VectorLoad<BLOCK_PER_THREADS>(frag_y, features_xj + decimal_y,
                                        frag_y_size, thread_id);
          __syncthreads();
          for (int i = thread_id; i < x_frag_size; i += blockDim.x) {
            float frag_x_value = frag_x[i];
            for (int j = 0; j < frag_y_size; ++j) {
              sub_kernel_sum += kernel_method(frag_x_value, frag_y[j]);
            }
          }
          __syncthreads();
        }

        sub_kernel_sum = block_reduction_sum(sub_kernel_sum);
        if (thread_id == 0) {
          float diff = (sub_kernel_sum / (static_cast<float>(len_xi * len_xj)));
          distance_matrix[xi * size + xj] = diff;
        }
      }
    }
  }

  // TODO: May need to refactor this messy code.
  template <typename BinaryOp>
  void __global__ MMDKernelCompute(
      float* x_features, float* y_features, int* x_offsets, int* y_offsets,
      int* x_sizes, int* y_sizes, int size_x, int size_y,
      float* distance_matrix, BinaryOp kernel_method) {
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    // __shared__ float sub_sum[BLOCK_PER_THREADS];
    __shared__ float frag_x[MTILE_SIZE];
    __shared__ float frag_y[NTILE_SIZE];
    float local_frag_y[THREADS_WORKING_SIZE];

    // divice workloads for every block
    int num_workloads = size_x * size_y;
    int workloads_per_block = (num_workloads + gridDim.x - 1) / gridDim.x;
    int start = workloads_per_block * block_id;
    int end = min(start + workloads_per_block, num_workloads);

    // const auto& x_features = *x_features_ptr;
    //  for (int xi = 0; xi < size; ++xi) {
    //  for (int xj = start; xj < end; ++xj) {
    for (int global_index = start; global_index < end; ++global_index) {
      int xi = global_index / size_y;
      int xj = global_index % size_y;
      int len_xi = x_sizes[xi];
      int len_xj = y_sizes[xj];
      float* features_xi = x_features + x_offsets[xi];
      float* features_xj = y_features + y_offsets[xj];

      float sub_kernel_sum = 0.0f;

      int decimal_x = roundUp(len_xi, MTILE_SIZE);
      int decimal_y = roundUp(len_xj, NTILE_SIZE);
      // decimal_x && decimal_y
      for (int x_frag_start = 0; x_frag_start < decimal_x;
           x_frag_start += MTILE_SIZE) {
        // load frag_x
        VectorLoad<BLOCK_PER_THREADS>(frag_x, features_xi + x_frag_start,
                                      MTILE_SIZE, thread_id);
        for (int y_frag_start = 0; y_frag_start < decimal_y;
             y_frag_start += NTILE_SIZE) {
          // Load frag_y
          VectorLoad<BLOCK_PER_THREADS>(frag_y, features_xj + y_frag_start,
                                        NTILE_SIZE, thread_id);
          __syncthreads();

#pragma unroll
          for (int j = 0; j < NTILE_SIZE; j += THREADS_WORKING_SIZE) {
#pragma unroll
            for (int sub_j = 0; sub_j < THREADS_WORKING_SIZE; ++sub_j) {
              local_frag_y[sub_j] = frag_y[j + sub_j];
            }
            for (int i = thread_id; i < MTILE_SIZE; i += blockDim.x) {
              for (int sub_j = 0; sub_j < THREADS_WORKING_SIZE; ++sub_j) {
                sub_kernel_sum += kernel_method(frag_x[i], local_frag_y[sub_j]);
              }
            }
          }
          __syncthreads();
        }
      }

      // decimal_x && tail_y
      {
        int frag_y_size = len_xj - decimal_y;
        VectorLoad<BLOCK_PER_THREADS>(frag_y, features_xj + decimal_y,
                                      frag_y_size, thread_id);
        // Tail loop of THREADS_WORKING_SIZE: to enable local_cache without if
        // branch
        int decimal_tail_y = roundUp(frag_y_size, THREADS_WORKING_SIZE);
        for (int x_frag_start = 0; x_frag_start < decimal_x;
             x_frag_start += MTILE_SIZE) {
          // load frag_x
          VectorLoad<BLOCK_PER_THREADS>(frag_x, features_xi + x_frag_start,
                                        MTILE_SIZE, thread_id);

          __syncthreads();
#pragma unroll
          for (int j = 0; j < decimal_tail_y; j += THREADS_WORKING_SIZE) {
#pragma unroll
            for (int sub_j = 0; sub_j < THREADS_WORKING_SIZE; ++sub_j) {
              local_frag_y[sub_j] = frag_y[j + sub_j];
            }
            for (int i = thread_id; i < MTILE_SIZE; i += blockDim.x) {
              for (int sub_j = 0; sub_j < THREADS_WORKING_SIZE; ++sub_j) {
                sub_kernel_sum += kernel_method(frag_x[i], local_frag_y[sub_j]);
              }
            }
          }

          // Tail loop of THREADS_WORKING_SIZE
          for (int i = thread_id; i < MTILE_SIZE; i += blockDim.x) {
            for (int j = decimal_tail_y; j < frag_y_size; ++j) {
              sub_kernel_sum += kernel_method(frag_x[i], frag_y[j]);
            }
          }
          __syncthreads();
        }
      }

      // decimal_y && tail_x
      {
        int x_frag_size = len_xi - decimal_x;
        VectorLoad<BLOCK_PER_THREADS>(frag_x, features_xi + decimal_x,
                                      x_frag_size, thread_id);
        for (int y_frag_start = 0; y_frag_start < decimal_y;
             y_frag_start += NTILE_SIZE) {
          // Load frag_y
          VectorLoad<BLOCK_PER_THREADS>(frag_y, features_xj + y_frag_start,
                                        NTILE_SIZE, thread_id);
          __syncthreads();

#pragma unroll
          for (int j = 0; j < NTILE_SIZE; j += THREADS_WORKING_SIZE) {
#pragma unroll
            for (int sub_j = 0; sub_j < THREADS_WORKING_SIZE; ++sub_j) {
              local_frag_y[sub_j] = frag_y[j + sub_j];
            }
            for (int i = thread_id; i < x_frag_size; i += blockDim.x) {
              for (int sub_j = 0; sub_j < THREADS_WORKING_SIZE; ++sub_j) {
                sub_kernel_sum += kernel_method(frag_x[i], local_frag_y[sub_j]);
              }
            }
          }
          __syncthreads();
        }
      }

      // tail_x && tail_y
      {
        // Tail of loop
        int x_frag_size = len_xi - decimal_x;
        int frag_y_size = len_xj - decimal_y;
        VectorLoad<BLOCK_PER_THREADS>(frag_y, features_xj + decimal_y,
                                      frag_y_size, thread_id);
        __syncthreads();

        for (int i = thread_id; i < x_frag_size; i += blockDim.x) {
          float frag_x_value = frag_x[i];
          for (int j = 0; j < frag_y_size; ++j) {
            sub_kernel_sum += kernel_method(frag_x_value, frag_y[j]);
          }
        }
        __syncthreads();
      }

      sub_kernel_sum = block_reduction_sum(sub_kernel_sum);
      if (thread_id == 0) {
        float diff = (sub_kernel_sum / (static_cast<float>(len_xi * len_xj)));
        distance_matrix[xj * size_x + xi] = diff;
      }
    }
  }
}  // namespace cpptools

#endif  // HPCORPIET_CPPTOOLS_MMD_CUDA_KERNEL_IMPL_H_
