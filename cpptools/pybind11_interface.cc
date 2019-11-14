//
// Created by king-kong on 23/10/19.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mmd_cpu.h"

PYBIND11_MODULE(cpptools, bind) {
  using Vec2D = std::vector<std::vector<double>>;
  bind.doc() = "CPU Computation Functions";
  //  bind.def("mmd_outerprod_linear",
  //  &cpptools::linear::MMDOuterProdGPU,
  //           "");
  // Below is some ugly static_cast to mitigate pybind11's inability to resolve
  // function overloading
  bind.def("mmd_self_linear",
           static_cast<Vec2D (*)(const Vec2D&)>(cpptools::linear::MMDSelf), "");
  bind.def(
      "mmd_linear",
      static_cast<Vec2D (*)(const Vec2D&, const Vec2D&)>(cpptools::linear::MMD),
      "");

  // bind.def("mmd_outerprod_rbf", &cpptools::rbf::MMDOuterProdGPU, "");
  bind.def("mmd_self_rbf",
           static_cast<Vec2D (*)(const Vec2D&, double)>(cpptools::rbf::MMDSelf),
           "");
  bind.def("mmd_rbf",
           static_cast<Vec2D (*)(const Vec2D&, const Vec2D&, double)>(
               cpptools::rbf::MMD),
           "");

  bind.def("mmd_self_rbf_mul", cpptools::rbf::MMDSelfMul);

  bind.def("mean_map_rbf", cpptools::rbf::MeanMapRBF);
}