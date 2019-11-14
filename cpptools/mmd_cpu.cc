#include "mmd_cpu.h"
#include <omp.h>
#include <cmath>
#include "mmd_cpu_impl.h"

namespace cpptools {
namespace linear {
std::vector<std::vector<float>> MMDSelf(
    const std::vector<std::vector<float>>& x_features) {
  return cpptools::MMDSelfCompute(
      x_features, [](float x, float y) -> float { return x * y; });
}

std::vector<std::vector<float>> MMD(
    const std::vector<std::vector<float>>& x_features,
    const std::vector<std::vector<float>>& y_features) {
  return cpptools::MMDCompute(x_features, y_features,
                              [](float x, float y) -> float { return x * y; });
}

std::vector<std::vector<double>> MMDSelf(
    const std::vector<std::vector<double>>& x_features) {
  return cpptools::MMDSelfCompute(
      x_features, [](double x, double y) -> double { return x * y; });
}

std::vector<std::vector<double>> MMD(
    const std::vector<std::vector<double>>& x_features,
    const std::vector<std::vector<double>>& y_features) {
  return cpptools::MMDCompute(
      x_features, y_features,
      [](double x, double y) -> double { return x * y; });
}
}  // namespace linear
}  // namespace cpptools

namespace cpptools {
namespace rbf {
std::vector<float> MMDOuterProd(
    const std::vector<std::vector<float>>& x_features, float gamma) {
  return cpptools::MMDOuterProdCompute(
      x_features, [gamma](float x, float y) -> float {
        return std::exp(-gamma * (x - y) * (x - y));
      });
}

std::vector<std::vector<float>> MMDSelf(
    const std::vector<std::vector<float>>& x_features, float gamma) {
  return cpptools::MMDSelfCompute(x_features,
                                  [gamma](float x, float y) -> float {
                                    return std::exp(-gamma * (x - y) * (x - y));
                                  });
}

std::vector<std::vector<float>> MMD(
    const std::vector<std::vector<float>>& x_features,
    const std::vector<std::vector<float>>& y_features, float gamma) {
  return cpptools::MMDCompute(x_features, y_features,
                              [gamma](float x, float y) -> float {
                                return std::exp(-gamma * (x - y) * (x - y));
                              });
}

std::vector<double> MMDOuterProd(
    const std::vector<std::vector<double>>& x_features, double gamma) {
  return cpptools::MMDOuterProdCompute(
      x_features, [gamma](double x, double y) -> double {
        return std::exp(-gamma * (x - y) * (x - y));
      });
}

std::vector<std::vector<double>> MMDSelf(
    const std::vector<std::vector<double>>& x_features, double gamma) {
  return cpptools::MMDSelfCompute(x_features,
                                  [gamma](double x, double y) -> double {
                                    return std::exp(-gamma * (x - y) * (x - y));
                                  });
}

std::vector<std::vector<double>> MMD(
    const std::vector<std::vector<double>>& x_features,
    const std::vector<std::vector<double>>& y_features, double gamma) {
  return cpptools::MMDCompute(x_features, y_features,
                              [gamma](double x, double y) -> double {
                                return std::exp(-gamma * (x - y) * (x - y));
                              });
}

std::vector<std::vector<double>> MeanMapRBF(
    const std::vector<std::vector<double>>& distance_matrix,
    const std::vector<double>& self_y, const std::vector<double>& self_x,
    double gamma) {
  auto row_size = distance_matrix.size();
  if (row_size <= 0) return std::vector<std::vector<double>>(0);
  auto col_size = distance_matrix.front().size();
  std::vector<std::vector<double>> embedding_matrix(
      row_size, std::vector<double>(col_size, 0.0));
  double base = std::exp(-gamma);

#pragma omp parallel for default(none)                           \
    shared(row_size, col_size, embedding_matrix, self_y, self_x, \
           distance_matrix, base)
  for (int i = 0; i < row_size; ++i) {
    for (int j = 0; j < col_size; ++j) {
      double square_distance =
          self_y[i] + self_x[j] - 2 * distance_matrix[i][j];
      embedding_matrix[i][j] = std::pow(base, square_distance);
    }
  }

  return embedding_matrix;
}

std::vector<double> MMDOuterProdMul(
    const std::vector<std::vector<std::vector<double>>>& x_features,
    double gamma) {
  auto size = x_features.size();
  const int k_feature_size = x_features.front().front().size();
  std::vector<double> x_outer_prod(size);

#pragma omp parallel for default(none) \
    shared(size, gamma, x_features, x_outer_prod)
  for (int index_i = 0; index_i < size; ++index_i) {
    auto len_index_i = x_features[index_i].size();
    double sum = 0.0;

    for (int i = 0; i < len_index_i; ++i) {
      for (int j = 0; j < len_index_i; ++j) {
        double sub_sum = 0.0;
        for (int k = 0; k < k_feature_size; ++k) {
          double tmp = (x_features[index_i][i][k] - x_features[index_i][j][k]);
          tmp = tmp * tmp;
          sub_sum += tmp;
        }
        sum += std::exp(-gamma * sub_sum);
      }
    }

    x_outer_prod[index_i] =
        sum / static_cast<double>(len_index_i * len_index_i);
  }

  return x_outer_prod;
}

// Reference implementation
std::vector<std::vector<double>> MMDSelfMul(
    const std::vector<std::vector<std::vector<double>>>& x_features,
    double gamma) {
  auto size = x_features.size();
  if (size <= 0) {
    return std::vector<std::vector<double>>(0);
  }
  if (x_features.front().size() <= 0) {
    return std::vector<std::vector<double>>(0);
  }

  std::vector<std::vector<double>> distance_matrix(
      size, std::vector<double>(size, 0.0));

  // size of feature should be consistent, I don't check it here.
  const int kFeatureSize = x_features.front().front().size();
#pragma omp parallel for default(none) \
    shared(size, gamma, distance_matrix, x_features)
  for (int index_i = 0; index_i < size; ++index_i) {
    for (int index_j = index_i; index_j < size; ++index_j) {
      double sum = 0.0;
      auto len_index_i = x_features[index_i].size();
      auto len_index_j = x_features[index_j].size();

      for (int i = 0; i < len_index_i; ++i) {
        for (int j = 0; j < len_index_j; ++j) {
          double sub_sum = 0.0;
          for (int k = 0; k < kFeatureSize; ++k) {
            double tmp =
                (x_features[index_i][i][k] - x_features[index_j][j][k]);
            tmp = tmp * tmp;
            sub_sum += tmp;
          }
          sum += std::exp(-gamma * sub_sum);
        }
      }

      distance_matrix[index_i][index_j] =
          sum / static_cast<double>(len_index_i * len_index_j);
    }
  }

  for (int i = 0; i < size; ++i) {
    for (int j = i + 1; j < size; ++j) {
      distance_matrix[j][i] = distance_matrix[i][j];
    }
  }
  return distance_matrix;
}

std::vector<std::vector<double>> MMDMul(
    const std::vector<std::vector<std::vector<double>>>& x_features,
    const std::vector<std::vector<std::vector<double>>>& y_features,
    double gamma) {
  auto size_x = x_features.size();
  // TODO: extract error checking
  if (size_x <= 0) {
    return std::vector<std::vector<double>>(0);
  }
  if (x_features.front().size() <= 0) {
    return std::vector<std::vector<double>>(0);
  }

  auto size_y = y_features.size();
  std::vector<std::vector<double>> distance_matrix(
      size_y, std::vector<double>(size_x, 0.0));

  // size_x of feature should be consistent, I don't check it here.
  const int kFeatureSize = x_features.front().front().size();
#pragma omp parallel for default(none) \
    shared(size_x, size_y, gamma, distance_matrix, x_features, y_features)
  for (int index_i = 0; index_i < size_y; ++index_i) {
    for (int index_j = 0; index_j < size_x; ++index_j) {
      double sum = 0.0;
      auto len_index_i = y_features[index_i].size();
      auto len_index_j = x_features[index_j].size();

      for (int i = 0; i < len_index_i; ++i) {
        for (int j = 0; j < len_index_j; ++j) {
          double sub_sum = 0.0;
          for (int k = 0; k < kFeatureSize; ++k) {
            double tmp =
                (y_features[index_i][i][k] - x_features[index_j][j][k]);
            tmp = tmp * tmp;
            sub_sum += tmp;
          }
          sum += std::exp(-gamma * sub_sum);
        }
      }

      distance_matrix[index_i][index_j] =
          sum / static_cast<double>(len_index_i * len_index_j);
    }
  }

  return distance_matrix;
}
}  // namespace rbf
}  // namespace cpptools

namespace cpptools {
namespace linear {
std::vector<std::vector<float>> MMDSelfOpt(
    const std::vector<std::vector<float>>& x_features) {
  return MMDSelfKernelComputeOpt(x_features);
}

std::vector<std::vector<double>> MMDSelfOpt(
    const std::vector<std::vector<double>>& x_features) {
  return MMDSelfKernelComputeOpt(x_features);
}

std::vector<std::vector<float>> MMDOpt(
    const std::vector<std::vector<float>>& x_features,
    const std::vector<std::vector<float>>& y_features) {
  return MMDKernelComputeOpt(x_features, y_features);
}

std::vector<std::vector<double>> MMDOpt(
    const std::vector<std::vector<double>>& x_features,
    const std::vector<std::vector<double>>& y_features) {
  return MMDKernelComputeOpt(x_features, y_features);
}
}  // namespace linear
}  // namespace cpptools