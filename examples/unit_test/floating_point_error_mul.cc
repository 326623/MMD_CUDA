//
// Created by king-kong on 29/10/19.
//
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "mmd_cpu.h"
#include "mmd_cuda.h"
namespace {
template <typename T>
using Vec3D = std::vector<std::vector<std::vector<T>>>;

template <typename T>
using Vec2D = std::vector<std::vector<T>>;

class MMDMulErrorTest : public ::testing::Test {
 public:
  Vec3D<float> x_features;
  Vec3D<float> y_features;
  int size_x;
  int size_y;
  static const int kFeatureSize = 2;

  MMDMulErrorTest() {
    // Perform setup here
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rand_real_num(-10.0, 10.0);
    std::uniform_int_distribution<int> rand_int_num(300, 3000);
    size_x = 16;
    size_y = 8;
    x_features = Vec3D<float>(size_x);
    y_features = Vec3D<float>(size_y);

    for (int i = 0; i < size_x; ++i) {
      x_features[i] = Vec2D<float>(rand_int_num(gen));
      std::generate(x_features[i].begin(), x_features[i].end(),
                    [&rand_real_num, &gen]() {
                      std::vector<float> tmp_vec(kFeatureSize);
                      for (int i = 0; i < kFeatureSize; ++i) {
                        tmp_vec[i] = rand_real_num(gen);
                      }
                      return tmp_vec;
                    });
    }

    for (int i = 0; i < size_y; ++i) {
      y_features[i] = Vec2D<float>(rand_int_num(gen));
      std::generate(y_features[i].begin(), y_features[i].end(),
                    [&rand_real_num, &gen]() {
                      std::vector<float> tmp_vec(kFeatureSize);
                      for (int i = 0; i < kFeatureSize; ++i) {
                        tmp_vec[i] = rand_real_num(gen);
                      }
                      return tmp_vec;
                    });
    }
  }
};

TEST_F(MMDMulErrorTest, MMDOuterProdCPUVSGPU) {
  // Transfer back
  Vec3D<double> double_x_features(size_x);
  for (int i = 0; i < size_x; ++i) {
    double_x_features[i].resize(x_features[i].size());
    for (int j = 0; j < double_x_features[i].size(); ++j) {
      double_x_features[i][j].resize(kFeatureSize);
      for (int k = 0; k < kFeatureSize; ++k) {
        double_x_features[i][j][k] = static_cast<double>(x_features[i][j][k]);
      }
    }
  }

  std::vector<double> x_outer_prod =
      cpptools::rbf::MMDOuterProdMul(double_x_features, 0.5);

  auto gpu_x_outer_prod = cpptools::rbf::MMDOuterProdGPUMul(x_features, 0.5f);

  for (int i = 0; i < size_x; ++i) {
    EXPECT_NEAR(static_cast<double>(x_outer_prod[i]), gpu_x_outer_prod[i], 1e-4)
        << "(" << i << ")";
  }
}

TEST_F(MMDMulErrorTest, MMDSelfCPUVSGPU) {
  // Transfer back
  Vec3D<double> double_x_features(size_x);
  for (int i = 0; i < size_x; ++i) {
    double_x_features[i].resize(x_features[i].size());
    for (int j = 0; j < double_x_features[i].size(); ++j) {
      double_x_features[i][j].resize(kFeatureSize);
      for (int k = 0; k < kFeatureSize; ++k) {
        double_x_features[i][j][k] = static_cast<double>(x_features[i][j][k]);
      }
    }
  }

  std::vector<std::vector<double>> cpu_distance_matrix =
      cpptools::rbf::MMDSelfMul(double_x_features, 0.5);

  auto gpu_distance_matrix = cpptools::rbf::MMDSelfGPUMul(x_features, 0.5f);

  for (int i = 0; i < size_x; ++i) {
    for (int j = 0; j < size_x; ++j) {
      EXPECT_NEAR(static_cast<double>(cpu_distance_matrix[i][j]),
                  gpu_distance_matrix[i][j], 1e-4)
          << "(" << i << ", " << j << ")";
    }
  }
}

TEST_F(MMDMulErrorTest, MMDCPUVSGPU) {
  // Transfer back
  Vec3D<double> double_x_features(size_x);
  for (int i = 0; i < size_x; ++i) {
    double_x_features[i].resize(x_features[i].size());
    for (int j = 0; j < double_x_features[i].size(); ++j) {
      double_x_features[i][j].resize(kFeatureSize);
      for (int k = 0; k < kFeatureSize; ++k) {
        double_x_features[i][j][k] = static_cast<double>(x_features[i][j][k]);
      }
    }
  }

  Vec3D<double> double_y_features(size_y);
  for (int i = 0; i < size_y; ++i) {
    double_y_features[i].resize(y_features[i].size());
    for (int j = 0; j < double_y_features[i].size(); ++j) {
      double_y_features[i][j].resize(kFeatureSize);
      for (int k = 0; k < kFeatureSize; ++k) {
        double_y_features[i][j][k] = static_cast<double>(y_features[i][j][k]);
      }
    }
  }

  std::vector<std::vector<double>> cpu_distance_matrix =
      cpptools::rbf::MMDMul(double_x_features, double_y_features, 0.5);

  auto gpu_distance_matrix =
      cpptools::rbf::MMDGPUMul(x_features, y_features, 0.5f);

  for (int i = 0; i < size_y; ++i) {
    for (int j = 0; j < size_x; ++j) {
      EXPECT_NEAR(static_cast<double>(cpu_distance_matrix[i][j]),
                  gpu_distance_matrix[i][j], 1e-4)
          << "(" << i << ", " << j << ")";
    }
  }
}
}  // namespace