//
// Created by king-kong on 29/10/19.
//
#include <benchmark/benchmark.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <array>
#include <iostream>
#include <random>
#include <vector>
#include "mmd_cuda.h"
#include "utils_cuda.h"

#define FEATURE_DIMENSION 20

class SetUpFixture {
  int64_t num_x_features_;
  int64_t feature_dim_;
  std::vector<std::vector<std::vector<float>>> x_features_;

 public:
  const std::vector<std::vector<std::vector<float>>>& GetXFeatures() const {
    return x_features_;
  }

 public:
  SetUpFixture(int64_t num_x_features, int64_t dim_feature)
      : num_x_features_(num_x_features),
        x_features_(num_x_features),
        feature_dim_(dim_feature) {
    // Perform setup here
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rand_real_num(-10.0, 10.0);
    std::uniform_int_distribution<int> rand_int_num(300, 3000);
    std::vector<int> sizes(num_x_features);
    const int feature_dim = feature_dim_;
    // roughly calculated
    for (int i = 0; i < num_x_features; ++i) {
      x_features_[i] = std::vector<std::vector<float>>(rand_int_num(gen));
      // std::vector<float> x_feature(rand_int_num(gen));
      std::generate(x_features_[i].begin(), x_features_[i].end(),
                    [&rand_real_num, &gen, feature_dim]() {
                      std::vector<float> gen_array(feature_dim);
                      for (int i = 0; i < feature_dim; ++i) {
                        gen_array[i] = rand_real_num(gen);
                      }
                      return gen_array;
                    });
    }
  }

  int64_t CalMMDSelfFlops() {
    int64_t num_flops = 0;
    for (int i = 0; i < num_x_features_; ++i) {
      for (int j = i; j < num_x_features_; ++j) {
        num_flops +=
            x_features_[i].size() * x_features_[j].size() * (feature_dim_ + 1);
      }
    }
    return num_flops;
  }

  int64_t CalMMDFlops() {
    int64_t num_flops = 0;

    for (int i = 0; i < num_x_features_; ++i) {
      for (int j = 0; j < num_x_features_; ++j) {
        num_flops +=
            x_features_[i].size() * x_features_[j].size() * (feature_dim_ + 1);
      }
    }
    return num_flops;
  }
};

static void BM_MMDSelfKernelSingleRBFMulKernel(benchmark::State& state) {
  int64_t num_x_features = state.range(0);
  SetUpFixture setup_fixture(num_x_features, FEATURE_DIMENSION);
  std::vector<std::vector<std::vector<float>>> x_features =
      setup_fixture.GetXFeatures();
  int64_t num_flops = setup_fixture.CalMMDSelfFlops();

  cudaProfilerStart();
  for (auto _ : state) {
    cpptools::rbf::MMDSelfGPUMul(x_features, 0.5f);
    cudaDeviceSynchronize();
  }
  cudaProfilerStop();

  state.SetItemsProcessed(int64_t(state.iterations() * num_flops));
}
//
// static void BM_MMDKernelSingleLinearKernel(benchmark::State& state) {
//  int64_t num_x_features = state.range(0);
//  SetUpFixture<FEATURE_DIMENSION> setup_fixture(num_x_features);
//  std::vector<std::vector<std::array<float, FEATURE_DIMENSION>>> x_features =
//      setup_fixture.GetXFeatures();
//  int64_t num_flops = setup_fixture.CalMMDFlops();
//
//  cudaProfilerStart();
//  for (auto _ : state) {
//    cpptools::linear::MMDGPUKernel(x_features, x_features);
//    cudaDeviceSynchronize();
//  }
//  cudaProfilerStop();
//
//  state.SetItemsProcessed(int64_t(state.iterations() * num_flops));
//}

// Register the function as a benchmark
// BENCHMARK(BM_SomeFunction)->Arg(8); //->RangeMultiplier(2)->Range(8, 1024);
BENCHMARK(BM_MMDSelfKernelSingleRBFMulKernel)
    ->RangeMultiplier(2)
    ->Range(8, 1024);
// BENCHMARK(BM_MMDKernelSingleLinearKernel)->RangeMultiplier(2)->Range(8,
// 2048); BENCHMARK(BM_SomeFunction)->RangeMultiplier(2)->Range(512, 1024);
