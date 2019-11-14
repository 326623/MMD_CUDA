//
// Created by king-kong on 14/10/19.
//
#include <benchmark/benchmark.h>
#include <mmd_cpu.h>
#include <random>

// static void BM_StringCreation(benchmark::State& state) {
//  for (auto _ : state)
//    std::string empty_string;
//}
//// Register the function as a benchmark
// BENCHMARK(BM_StringCreation)->Arg(8);

namespace {
void SetUp(int64_t num_x_features, std::vector<std::vector<double>>& x_features,
           int64_t& num_flops) {
  num_flops = 0;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> rand_real_num(-10.0, 10.0);
  std::uniform_int_distribution<int> rand_int_num(300, 3000);
  // roughly calculated
  for (int i = 0; i < num_x_features; ++i) {
    std::vector<double> x_feature(rand_int_num(gen));
    std::generate(x_feature.begin(), x_feature.end(),
                  [&rand_real_num, &gen]() { return rand_real_num(gen); });
    x_features.emplace_back(x_feature);
  }

  // outer(x_i, x_i)
  for (int i = 0; i < num_x_features; ++i) {
    num_flops += x_features[i].size() * x_features[i].size();
  }

  for (int i = 0; i < num_x_features; ++i) {
    for (int j = i + 1; j < num_x_features; ++j) {
      num_flops += x_features[i].size() * x_features[j].size();
    }
  }
}

template <typename Float>
static void BenchMMDSelfFloat(::benchmark::State& state) {
  std::vector<std::vector<double>> x_features;
  int64_t num_flops;
  SetUp(state.range(0), x_features, num_flops);
  for (auto _ : state) {
    auto matrix = cpptools::linear::MMDSelf(x_features);
  }
  state.SetItemsProcessed(int64_t(state.iterations() * num_flops));
}

template <typename Float>
static void BenchMMDFloat(benchmark::State& state) {
  std::vector<std::vector<double>> x_features;
  int64_t num_flops;
  SetUp(state.range(0), x_features, num_flops);

  for (auto _ : state) {
    auto matrix = cpptools::linear::MMD(x_features, x_features);
  }
  //  state.SetBytesProcessed(
  //      int64_t(state.iterations() * num_flops * sizeof(double)));
  state.SetItemsProcessed(int64_t(state.iterations() * num_flops));
}
// This is the scale we are looking at. 4096 number of time-series data points.
BENCHMARK_TEMPLATE(BenchMMDSelfFloat, float)->RangeMultiplier(2)->Range(8, 1024);
BENCHMARK_TEMPLATE(BenchMMDFloat, float)->RangeMultiplier(2)->Range(8, 1024);
BENCHMARK_TEMPLATE(BenchMMDSelfFloat, double)->RangeMultiplier(2)->Range(8, 1024);
BENCHMARK_TEMPLATE(BenchMMDFloat, double)->RangeMultiplier(2)->Range(8, 1024);
}  // namespace

BENCHMARK_MAIN();
