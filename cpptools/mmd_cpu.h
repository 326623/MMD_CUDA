//
// Created by king-kong on 14/10/19.
//

#ifndef HPCORPIET_CPPTOOLS_MMD_CPU_H_
#define HPCORPIET_CPPTOOLS_MMD_CPU_H_
#include <vector>

namespace cpptools {
namespace linear {
std::vector<std::vector<float>> MMDSelf(
    const std::vector<std::vector<float>>& x_features);

std::vector<std::vector<float>> MMD(
    const std::vector<std::vector<float>>& x_features,
    const std::vector<std::vector<float>>& y_features);

std::vector<std::vector<double>> MMDSelf(
    const std::vector<std::vector<double>>& x_features);

std::vector<std::vector<double>> MMD(
    const std::vector<std::vector<double>>& x_features,
    const std::vector<std::vector<double>>& y_features);

std::vector<std::vector<float>> MMDSelfOpt(
    const std::vector<std::vector<float>>& x_features);

std::vector<std::vector<double>> MMDSelfOpt(
    const std::vector<std::vector<double>>& x_features);

std::vector<std::vector<float>> MMDOpt(
    const std::vector<std::vector<float>>& x_features,
    const std::vector<std::vector<float>>& y_features);

std::vector<std::vector<double>> MMDOpt(
    const std::vector<std::vector<double>>& x_features,
    const std::vector<std::vector<double>>& y_features);
}  // namespace linear
}  // namespace cpptools

namespace cpptools {
namespace rbf {
std::vector<float> MMDOuterProd(
    const std::vector<std::vector<float>>& x_features, float gamma);

std::vector<std::vector<float>> MMDSelf(
    const std::vector<std::vector<float>>& x_features, float gamma);

std::vector<std::vector<float>> MMD(
    const std::vector<std::vector<float>>& x_features,
    const std::vector<std::vector<float>>& y_features, float gamma);

std::vector<double> MMDOuterProd(
    const std::vector<std::vector<double>>& x_features, double gamma);

std::vector<std::vector<double>> MMDSelf(
    const std::vector<std::vector<double>>& x_features, double gamma);

std::vector<std::vector<double>> MMD(
    const std::vector<std::vector<double>>& x_features,
    const std::vector<std::vector<double>>& y_features, double gamma);

std::vector<double> MMDOuterProdMul(
    const std::vector<std::vector<std::vector<double>>>& x_features,
    double gamma);

std::vector<std::vector<double>> MMDSelfMul(
    const std::vector<std::vector<std::vector<double>>>& x_features,
    double gamma);

std::vector<std::vector<double>> MMDMul(
    const std::vector<std::vector<std::vector<double>>>& x_features,
    const std::vector<std::vector<std::vector<double>>>& y_features,
    double gamma);

std::vector<std::vector<double>> MeanMapRBF(
    const std::vector<std::vector<double>>& distance_matrix,
    const std::vector<double>& self_y, const std::vector<double>& self_x,
    double gamma);
}  // namespace rbf
}  // namespace cpptools

#endif  // HPCORPIET_SRC_MMD_CPU_H_
