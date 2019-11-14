//
// Created by king-kong on 2019/10/23.
//

#ifndef HPCORPIET_CPPTOOLS_MMD_CPU_IMPL_H_
#define HPCORPIET_CPPTOOLS_MMD_CPU_IMPL_H_

#include <omp.h>
#include <tqdm.h>
#include <cmath>
#include <vector>
#endif  // HPCORPIET_CPPTOOLS_MMD_CPU_IMPL_H_
namespace cpptools {
template <typename BinaryOp, typename Float>
std::vector<Float> MMDOuterProdCompute(
    const std::vector<std::vector<Float>>& x_features,
    BinaryOp kernel_function) {
  std::vector<Float> x_outer_prod(x_features.size());

  for (int i = 0; i < x_features.size(); ++i) {
    Float sum = Float{0.0};
    for (Float x : x_features[i]) {
      for (Float y : x_features[i]) {
        sum += kernel_function(x, y);
      }
    }
    x_outer_prod[i] =
        sum / static_cast<Float>(x_features[i].size() * x_features[i].size());
  }

  return x_outer_prod;
}

template <typename BinaryOp, typename Float>
std::vector<std::vector<Float>> MMDSelfCompute(
    const std::vector<std::vector<Float>>& x_features,
    BinaryOp kernel_function) {
  std::vector<std::vector<Float>> matrix(
      x_features.size(), std::vector<Float>(x_features.size(), Float(0.0)));

  tqdm progress_bar;
  int progress_counter = 0;
#pragma omp parallel for default(none) shared( \
    x_features, matrix, progress_counter, kernel_function, progress_bar)
  for (int i = 0; i < x_features.size(); ++i) {
    int tid = omp_get_thread_num();
#pragma omp atomic
    ++progress_counter;

    if (tid == 0) {
      progress_bar.progress(progress_counter,
                            static_cast<int>(x_features.size()));
    }
    for (int j = i; j < x_features.size(); ++j) {
      Float diff = Float(0.0);
      Float sum_kernel = Float(0.0);

      for (const auto& feature1 : x_features[i]) {
        for (const auto& feature2 : x_features[j]) {
          sum_kernel += kernel_function(feature1, feature2);
        }
      }

      diff = sum_kernel /
             static_cast<Float>(x_features[i].size() * x_features[j].size());

      matrix[i][j] = diff;
    }
  }

  for (int i = 0; i < x_features.size(); ++i) {
    for (int j = i + 1; j < x_features.size(); ++j) {
      matrix[j][i] = matrix[i][j];
    }
  }
  return matrix;
}

template <typename BinaryOp, typename Float>
std::vector<std::vector<Float>> MMDCompute(
    const std::vector<std::vector<Float>>& x_features,
    const std::vector<std::vector<Float>>& y_features,
    BinaryOp kernel_function) {
  std::vector<std::vector<Float>> matrix(
      y_features.size(), std::vector<Float>(x_features.size(), Float(0.0)));

  tqdm progress_bar;
  int progress_counter = 0;
#pragma omp parallel for default(none)                                    \
    shared(x_features, y_features, matrix, progress_bar, kernel_function, \
           progress_counter)
  for (int i = 0; i < y_features.size(); ++i) {
    int tid = omp_get_thread_num();
#pragma omp atomic
    ++progress_counter;

    if (tid == 0) {
      progress_bar.progress(progress_counter,
                            static_cast<int>(y_features.size()));
    }
    for (int j = 0; j < x_features.size(); ++j) {
      Float diff = Float(0.0);
      Float sum_kernel = Float(0.0);
      for (const auto& feature1 : x_features[j]) {
        for (const auto& feature2 : y_features[i]) {
          sum_kernel += kernel_function(feature1, feature2);
        }
      }
      diff = (sum_kernel /
              static_cast<Float>(x_features[j].size() * y_features[i].size()));

      matrix[i][j] = diff;
    }
  }
  return matrix;
}
}  // namespace cpptools

namespace cpptools {
namespace linear {
template <typename Float>
std::vector<std::vector<Float>> MMDSelfKernelComputeOpt(
    const std::vector<std::vector<Float>>& x_features) {
  std::vector<std::vector<Float>> distance_matrix(
      x_features.size(), std::vector<Float>(x_features.size(), 0.0f));
  std::vector<Float> sum_of_vector(x_features.size(), 0.0f);
  for (int i = 0; i < x_features.size(); ++i) {
    for (Float num : x_features[i]) {
      sum_of_vector[i] += num;
    }
    sum_of_vector[i] /= static_cast<Float>(x_features[i].size());
  }

  for (int i = 0; i < sum_of_vector.size(); ++i) {
    for (int j = i; j < sum_of_vector.size(); ++j) {
      distance_matrix[i][j] = sum_of_vector[i] * sum_of_vector[j];
    }
  }

  for (int i = 0; i < sum_of_vector.size(); ++i) {
    for (int j = i + 1; j < sum_of_vector.size(); ++j) {
      distance_matrix[j][i] = distance_matrix[i][j];
    }
  }
  return distance_matrix;
}

template <typename Float>
std::vector<std::vector<Float>> MMDKernelComputeOpt(
    const std::vector<std::vector<Float>>& x_features,
    const std::vector<std::vector<Float>>& y_features) {
  std::vector<std::vector<Float>> distance_matrix(
      x_features.size(), std::vector<Float>(x_features.size(), 0.0f));
  std::vector<Float> sum_of_x_features(x_features.size(), 0.0f);
  for (int i = 0; i < x_features.size(); ++i) {
    for (Float num : x_features[i]) {
      sum_of_x_features[i] += num;
    }
    sum_of_x_features[i] /= static_cast<Float>(x_features[i].size());
  }

  std::vector<Float> sum_of_y_features(y_features.size(), 0.0f);
  for (int i = 0; i < y_features.size(); ++i) {
    for (Float num : y_features[i]) {
      sum_of_y_features[i] += num;
    }
    sum_of_y_features[i] /= static_cast<Float>(y_features[i].size());
  }

  for (int i = 0; i < sum_of_y_features.size(); ++i) {
    for (int j = 0; j < sum_of_x_features.size(); ++j) {
      distance_matrix[i][j] = sum_of_y_features[i] * sum_of_x_features[j];
    }
  }
  return distance_matrix;
}
}  // namespace linear
}  // namespace cpptools
