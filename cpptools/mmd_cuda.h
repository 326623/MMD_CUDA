#ifndef HPCORPIET_CPPTOOLS_MMD_CUDA_CUH_
#define HPCORPIET_CPPTOOLS_MMD_CUDA_CUH_
#include <array>
#include <vector>

namespace cpptools {
namespace linear {
std::vector<float> MMDOuterProdGPU(
    const std::vector<std::vector<float>>& x_features);

std::vector<std::vector<float>> MMDSelfGPU(
    const std::vector<std::vector<float>>& x_features);

std::vector<std::vector<float>> MMDGPU(
    const std::vector<std::vector<float>>& x_features,
    const std::vector<std::vector<float>>& y_features);

// The following function is for internal usage only. The pointer passed here
// should point to GPU
}  // namespace linear

namespace rbf {
std::vector<float> MMDOuterProdGPU(
    const std::vector<std::vector<float>>& x_features, float gamma);

std::vector<std::vector<float>> MMDSelfGPU(
    const std::vector<std::vector<float>>& x_features, float gamma);

std::vector<std::vector<float>> MMDGPU(
    const std::vector<std::vector<float>>& x_features,
    const std::vector<std::vector<float>>& y_features, float gamma);

std::vector<float> MMDOuterProdGPUMul(
    const std::vector<std::vector<std::vector<float>>>& x_features,
    float gamma);

std::vector<std::vector<float>> MMDSelfGPUMul(
    const std::vector<std::vector<std::vector<float>>>& x_features,
    float gamma);

std::vector<std::vector<float>> MMDGPUMul(
    const std::vector<std::vector<std::vector<float>>>& x_features,
    const std::vector<std::vector<std::vector<float>>>& y_features,
    float gamma);
}  // namespace rbf

namespace experimental {
namespace rbf {
std::vector<std::vector<float>> MMDSelfGPUKernel(
    const std::vector<std::vector<float>>& x_features, float gamma);
}  // namespace rbf
}  // namespace experimental

// Assume distance_matrix are sqaure matrix
// Given upper triangle matrix, fill its lower triangle.
void TransposeFill(float** distance_matrix, int size);

void TransposeFill(float* distance_matrix, int size);
// void __global__ TransposeFillKernel(float** distance_matrix, int size);
}  // namespace cpptools
#endif