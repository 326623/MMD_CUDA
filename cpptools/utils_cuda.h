//
// Created by king-kong on 18/10/19.
//

#ifndef HPCORPIET_CPPTOOLS_UTILS_CUDA_H_
#define HPCORPIET_CPPTOOLS_UTILS_CUDA_H_
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <tuple>
#if defined(NDEBUG)
#define DCHECK_IS_ON() 0
#else
#define DCHECK_IS_ON() 1
#endif

// This module contains the GPU memory allocation and destruction. CudaPtr is
// basically an implementation of unique_ptr of GPU.

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(0);                                                 \
    }                                                          \
  }

#if DCHECK_IS_ON()
#define DCudaCheckError()                                      \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(0);                                                 \
    }                                                          \
  }
#else
#define DCudaCheckError() \
  {                       \
    do {                  \
    } while (0);          \
  }
#endif

template <typename T>
class CudaPtr {
 private:
  T* device_ptr_;

 public:
  CudaPtr() = delete;
  CudaPtr(const CudaPtr<T>& other) = delete;

  explicit CudaPtr(int size) : device_ptr_(nullptr) {
    cudaMalloc((void**)&device_ptr_, size * sizeof(T));
    DCudaCheckError();
  }

  CudaPtr(CudaPtr<T>&& other) : device_ptr_(other.device_ptr_) {
    other.device_ptr_ = nullptr;
  }

  void CopyToGPU(const T* host_data, std::size_t size) const {
    cudaMemcpy(device_ptr_, host_data, size * sizeof(T),
               cudaMemcpyHostToDevice);
    DCudaCheckError();
  }

  void CopyToCPU(T* host_data, std::size_t size) const {
    cudaMemcpy(host_data, device_ptr_, size * sizeof(T),
               cudaMemcpyDeviceToHost);
    DCudaCheckError();
  }

  void CopyToGPU(const std::vector<T>& host_vec) {
    CopyToGPU(host_vec.data(), host_vec.size());
  }

  void CopyToCPU(std::vector<T>& host_vec) {
    CopyToCPU(host_vec.data(), host_vec.size());
  }

  T* rawPtr() const { return device_ptr_; }

  ~CudaPtr() {
    if (device_ptr_ != nullptr) {
      cudaFree(device_ptr_);
      DCudaCheckError();
    }
  }
};

using GPUFlatVec = std::tuple<CudaPtr<float>, CudaPtr<int>, CudaPtr<int>>;

GPUFlatVec ToFlatGPUMem(const std::vector<std::vector<float>>& features);

GPUFlatVec ToFlatGPUMem(const std::vector<std::vector<std::vector<float>>>& features);

std::vector<std::vector<float>> To2DVec(CudaPtr<float> device_distance_matrix,
                                        int row_size, int col_size);
#endif  // HPCORPIET_CPPTOOLS_UTILS_CUDA_H_
