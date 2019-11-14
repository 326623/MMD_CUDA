//
// Created by king-kong on 15/10/19.
//
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "mmd_cpu.h"
#include "mmd_cuda.h"

namespace {
class MMDErrorTest : public ::testing::Test {
 public:
  // Due to the implementation limitation of Cuda
  // https://github.com/thrust/thrust/issues/726
  std::vector<std::vector<float>> x_features;
  std::vector<std::vector<float>> y_features;
  int size_x;
  int size_y;

  MMDErrorTest() {
    // Perform setup here
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> rand_real_num(-10.0, 10.0);
    std::uniform_int_distribution<int> rand_int_num(300, 3000);
    size_x = 16;
    size_y = 8;
    x_features = std::vector<std::vector<float>>(size_x);
    y_features = std::vector<std::vector<float>>(size_y);
    for (int i = 0; i < size_x; ++i) {
      x_features[i] = std::vector<float>(rand_int_num(gen));
      std::generate(x_features[i].begin(), x_features[i].end(),
                    [&rand_real_num, &gen]() { return rand_real_num(gen); });
    }

    for (int i = 0; i < size_y; ++i) {
      y_features[i] = std::vector<float>(rand_int_num(gen));
      std::generate(y_features[i].begin(), y_features[i].end(),
                    [&rand_real_num, &gen]() { return rand_real_num(gen); });
    }
  }
};

TEST_F(MMDErrorTest, MMDSelfCudaVSCPUDoubleLinear) {
  std::vector<std::vector<float>> cuda_distance_matrix =
      cpptools::linear::MMDSelfGPU(x_features);

  std::vector<std::vector<double>> double_x_features(size_x);
  for (int i = 0; i < size_x; ++i) {
    double_x_features[i].resize(x_features[i].size());
    for (int j = 0; j < double_x_features[i].size(); ++j) {
      double_x_features[i][j] = static_cast<double>(x_features[i][j]);
    }
  }

  auto cpu_distance_matrix = cpptools::linear::MMDSelf(double_x_features);

  for (int i = 0; i < size_x; ++i) {
    for (int j = 0; j < size_x; ++j) {
      EXPECT_NEAR(static_cast<double>(cuda_distance_matrix[i][j]),
                  cpu_distance_matrix[i][j], 1e-4)
          << "(" << i << ", " << j << ")";
      //      EXPECT_DOUBLE_EQ(static_cast<double>(cuda_distance_matrix[i][j]),
      //                       cpu_distance_matrix[i][j]);
    }
  }
}

TEST_F(MMDErrorTest, MMDSelfCudaVSCPUSingleLinear) {
  std::vector<std::vector<float>> cuda_distance_matrix =
      cpptools::linear::MMDSelfGPU(x_features);

  auto cpu_distance_matrix = cpptools::linear::MMDSelf(x_features);

  for (int i = 0; i < size_x; ++i) {
    for (int j = 0; j < size_x; ++j) {
      EXPECT_NEAR((cuda_distance_matrix[i][j]), cpu_distance_matrix[i][j], 1e-4)
          << "(" << i << ", " << j << ")";
      //      EXPECT_FLOAT_EQ(static_cast<double>(cuda_distance_matrix[i][j]),
      //                       cpu_distance_matrix[i][j]);
    }
  }
}

TEST_F(MMDErrorTest, MMDCudaVSCPUDoubleLinear) {
  // Transfer back
  std::vector<std::vector<float>> cuda_distance_matrix =
      cpptools::linear::MMDGPU(x_features, y_features);

  std::vector<std::vector<double>> double_x_features(size_x);
  for (int i = 0; i < size_x; ++i) {
    double_x_features[i].resize(x_features[i].size());
    for (int j = 0; j < double_x_features[i].size(); ++j) {
      double_x_features[i][j] = static_cast<double>(x_features[i][j]);
    }
  }

  std::vector<std::vector<double>> double_y_features(size_y);
  for (int i = 0; i < size_y; ++i) {
    double_y_features[i].resize(y_features[i].size());
    for (int j = 0; j < double_y_features[i].size(); ++j) {
      double_y_features[i][j] = static_cast<double>(y_features[i][j]);
    }
  }

  auto cpu_distance_matrix =
      cpptools::linear::MMD(double_x_features, double_y_features);

  for (int i = 0; i < size_y; ++i) {
    for (int j = 0; j < size_x; ++j) {
      EXPECT_NEAR(static_cast<double>(cuda_distance_matrix[i][j]),
                  cpu_distance_matrix[i][j], 1e-4)
          << "(" << i << ", " << j << ")";
      //      EXPECT_DOUBLE_EQ(static_cast<double>(cuda_distance_matrix[i][j]),
      //                       cpu_distance_matrix[i][j]);
    }
  }
}

TEST_F(MMDErrorTest, MMDCudaVSCPUSingleLinear) {
  std::vector<std::vector<float>> cuda_distance_matrix =
      cpptools::linear::MMDGPU(x_features, y_features);

  auto cpu_distance_matrix = cpptools::linear::MMD(x_features, y_features);

  for (int i = 0; i < size_y; ++i) {
    for (int j = 0; j < size_x; ++j) {
      EXPECT_NEAR((cuda_distance_matrix[i][j]), cpu_distance_matrix[i][j], 1e-4)
          << "(" << i << ", " << j << ")";
      //      EXPECT_FLOAT_EQ(static_cast<double>(cuda_distance_matrix[i][j]),
      //                       cpu_distance_matrix[i][j]);
    }
  }
}

TEST_F(MMDErrorTest, MMDSelfCudaVSCPUDoubleRbf) {
  // Transfer back
  std::vector<std::vector<float>> cuda_distance_matrix =
      cpptools::rbf::MMDSelfGPU(x_features, 0.5f);

  std::vector<std::vector<double>> double_x_features(size_x);
  for (int i = 0; i < size_x; ++i) {
    double_x_features[i].resize(x_features[i].size());
    for (int j = 0; j < double_x_features[i].size(); ++j) {
      double_x_features[i][j] = static_cast<double>(x_features[i][j]);
    }
  }

  auto cpu_distance_matrix = cpptools::rbf::MMDSelf(double_x_features, 0.5);

  for (int i = 0; i < size_x; ++i) {
    for (int j = 0; j < size_x; ++j) {
      EXPECT_NEAR(static_cast<double>(cuda_distance_matrix[i][j]),
                  cpu_distance_matrix[i][j], 1e-4)
          << "(" << i << ", " << j << ")";
      //      EXPECT_DOUBLE_EQ(static_cast<double>(cuda_distance_matrix[i][j]),
      //                       cpu_distance_matrix[i][j]);
    }
  }
}

TEST_F(MMDErrorTest, MMDCudaVSCPUDoubleRbf) {
  // Transfer back
  std::vector<std::vector<float>> cuda_distance_matrix =
      cpptools::rbf::MMDGPU(x_features, y_features, 0.5f);

  std::vector<std::vector<double>> double_x_features(size_x);
  for (int i = 0; i < size_x; ++i) {
    double_x_features[i].resize(x_features[i].size());
    for (int j = 0; j < double_x_features[i].size(); ++j) {
      double_x_features[i][j] = static_cast<double>(x_features[i][j]);
    }
  }

  std::vector<std::vector<double>> double_y_features(size_y);
  for (int i = 0; i < size_y; ++i) {
    double_y_features[i].resize(y_features[i].size());
    for (int j = 0; j < double_y_features[i].size(); ++j) {
      double_y_features[i][j] = static_cast<double>(y_features[i][j]);
    }
  }

  auto cpu_distance_matrix =
      cpptools::rbf::MMD(double_x_features, double_y_features, 0.5);

  for (int i = 0; i < size_y; ++i) {
    for (int j = 0; j < size_x; ++j) {
      EXPECT_NEAR(static_cast<double>(cuda_distance_matrix[i][j]),
                  cpu_distance_matrix[i][j], 1e-4)
          << "(" << i << ", " << j << ")";
      //      EXPECT_DOUBLE_EQ(static_cast<double>(cuda_distance_matrix[i][j]),
      //                       cpu_distance_matrix[i][j]);
    }
  }
}

TEST_F(MMDErrorTest, MMDSelfCPUVSCPUOpt) {
  // Transfer back
  std::vector<std::vector<double>> double_x_features(size_x);
  for (int i = 0; i < size_x; ++i) {
    double_x_features[i].resize(x_features[i].size());
    for (int j = 0; j < double_x_features[i].size(); ++j) {
      double_x_features[i][j] = static_cast<double>(x_features[i][j]);
    }
  }

  std::vector<std::vector<double>> cpu_opt_distance_matrix =
      cpptools::linear::MMDSelfOpt(double_x_features);

  auto cpu_distance_matrix = cpptools::linear::MMDSelf(double_x_features);

  for (int i = 0; i < size_x; ++i) {
    for (int j = 0; j < size_x; ++j) {
      EXPECT_NEAR(static_cast<double>(cpu_opt_distance_matrix[i][j]),
                  cpu_distance_matrix[i][j], 1e-4)
          << "(" << i << ", " << j << ")";
      //      EXPECT_DOUBLE_EQ(static_cast<double>(cuda_distance_matrix[i][j]),
      //                       cpu_distance_matrix[i][j]);
    }
  }
}

TEST_F(MMDErrorTest, MMDCPUVSCPUOpt) {
  // Transfer back
  std::vector<std::vector<double>> double_x_features(size_x);
  for (int i = 0; i < size_x; ++i) {
    double_x_features[i].resize(x_features[i].size());
    for (int j = 0; j < double_x_features[i].size(); ++j) {
      double_x_features[i][j] = static_cast<double>(x_features[i][j]);
    }
  }

  std::vector<std::vector<double>> double_y_features(size_y);
  for (int i = 0; i < size_y; ++i) {
    double_y_features[i].resize(y_features[i].size());
    for (int j = 0; j < double_y_features[i].size(); ++j) {
      double_y_features[i][j] = static_cast<double>(y_features[i][j]);
    }
  }

  std::vector<std::vector<double>> cpu_opt_distance_matrix =
      cpptools::linear::MMDOpt(double_x_features, double_y_features);

  auto cpu_distance_matrix =
      cpptools::linear::MMD(double_x_features, double_y_features);

  for (int i = 0; i < size_y; ++i) {
    for (int j = 0; j < size_x; ++j) {
      EXPECT_NEAR(static_cast<double>(cpu_opt_distance_matrix[i][j]),
                  cpu_distance_matrix[i][j], 1e-4)
          << "(" << i << ", " << j << ")";
      //      EXPECT_DOUBLE_EQ(static_cast<double>(cuda_distance_matrix[i][j]),
      //                       cpu_distance_matrix[i][j]);
    }
  }
}

TEST_F(MMDErrorTest, OuterProd_CPU_GPU) {
  // Transfer back
  std::vector<std::vector<double>> double_x_features(size_x);
  for (int i = 0; i < size_x; ++i) {
    double_x_features[i].resize(x_features[i].size());
    for (int j = 0; j < double_x_features[i].size(); ++j) {
      double_x_features[i][j] = static_cast<double>(x_features[i][j]);
    }
  }

  std::vector<double> cpu_x_outer_prod =
      cpptools::rbf::MMDOuterProd(double_x_features, 0.5);

  auto gpu_distance_matrix = cpptools::rbf::MMDOuterProdGPU(x_features, 0.5);

  for (int i = 0; i < size_x; ++i) {
    EXPECT_NEAR(static_cast<double>(gpu_distance_matrix[i]),
                cpu_x_outer_prod[i], 1e-4)
        << "(" << i << ")";
  }
}

// The below three tests are expected to fail
// TEST_F(MMDErrorTest, MMDSelfCudaVSCPUSingleRbf) {
//  std::vector<std::vector<float>> cuda_distance_matrix =
//      cpptools::rbf::MMDSelfGPUKernel(x_features, 0.5f);
//
//  auto cpu_distance_matrix = cpptools::rbf::MMDSelfKernel(x_features, 0.5f);
//
//  for (int i = 0; i < size_x; ++i) {
//    for (int j = 0; j < size_x; ++j) {
//      EXPECT_NEAR((cuda_distance_matrix[i][j]), cpu_distance_matrix[i][j],
//      1e-4)
//                << "(" << i << ", " << j << ")";
//      //      EXPECT_FLOAT_EQ(static_cast<double>(cuda_distance_matrix[i][j]),
//      //                       cpu_distance_matrix[i][j]);
//    }
//  }
//}
//
// TEST_F(MMDErrorTest, MMDCudaVSCPUSingleRbf) {
//  // Transfer back
//  std::vector<std::vector<float>> cuda_distance_matrix =
//      cpptools::rbf::MMDGPUKernel(x_features, y_features, 0.5f);
//
//  std::vector<std::vector<float>> cpu_distance_matrix =
//      cpptools::rbf::MMDKernel(x_features, y_features, 0.5f);
//  // Strange numerical difference between single and double precision of CPU.
//  // While Cuda don't seem to have such issues.
//  /* std::vector<std::vector<double>> double_x_features(size);
//   for (int i = 0; i < size; ++i) {
//     double_x_features[i].resize(x_features[i].size());
//     for (int j = 0; j < double_x_features[i].size(); ++j) {
//       double_x_features[i][j] = static_cast<double>(x_features[i][j]);
//     }
//   }
//
//   std::vector<std::vector<double>> double_cpu_distance_matrix =
//   cpptools::MMDKernel( double_x_features, double_x_features, [](double x,
//   double y) -> double { return exp(-0.5 * (x - y) * (x - y));
//       });*/
//
//  for (int i = 0; i < size_y; ++i) {
//    for (int j = 0; j < size_x; ++j) {
//      EXPECT_NEAR((cuda_distance_matrix[i][j]), cpu_distance_matrix[i][j],
//      1e-4)
//          << "(" << i << ", " << j << ")";
//    }
//  }
//}
//
// TEST_F(MMDErrorTest, MMDCPUDoubleVSCPUSingleRbf) {
//  std::vector<std::vector<double>> double_x_features(size_x);
//  for (int i = 0; i < size_x; ++i) {
//    double_x_features[i].resize(x_features[i].size());
//    for (int j = 0; j < x_features[i].size(); ++j) {
//      double_x_features[i][j] = static_cast<double>(x_features[i][j]);
//    }
//  }
//
//  std::vector<std::vector<double>> double_y_features(size_y);
//  for (int i = 0; i < size_y; ++i) {
//    double_y_features[i].resize(y_features[i].size());
//    for (int j = 0; j < y_features[i].size(); ++j) {
//      double_y_features[i][j] = static_cast<double>(y_features[i][j]);
//    }
//  }
//
//  std::vector<std::vector<double>> double_cpu_distance_matrix =
//      cpptools::rbf::MMDKernel(double_x_features, double_y_features, 0.5);
//
//  std::vector<std::vector<float>> cpu_distance_matrix =
//      cpptools::rbf::MMDKernel(x_features, y_features, 0.5f);
//
//  for (int i = 0; i < size_y; ++i) {
//    for (int j = 0; j < size_x; ++j) {
//      EXPECT_NEAR((double_cpu_distance_matrix[i][j]),
//      cpu_distance_matrix[i][j],
//                  1e-4)
//          << "(" << i << ", " << j << ")";
//    }
//  }
//}
}  // namespace
