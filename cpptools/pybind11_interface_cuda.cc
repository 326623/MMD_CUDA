//
// Created by king-kong on 23/10/19.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mmd_cuda.h"

PYBIND11_MODULE(cudatools, bind) {
  bind.doc() = "CUDA Computation Functions";
  bind.def("mmd_outerprod_linear", &cpptools::linear::MMDOuterProdGPU,
           "");
  bind.def("mmd_self_linear", &cpptools::linear::MMDSelfGPU, "");
  bind.def("mmd_linear", &cpptools::linear::MMDGPU, "");

  bind.def("mmd_outerprod_rbf", &cpptools::rbf::MMDOuterProdGPU, "");
  bind.def("mmd_self_rbf", &cpptools::rbf::MMDSelfGPU, "");
  bind.def("mmd_rbf", &cpptools::rbf::MMDGPU, "");

  bind.def("mmd_outerprod_rbf_mul", &cpptools::rbf::MMDOuterProdGPUMul);
  bind.def("mmd_self_rbf_mul", &cpptools::rbf::MMDSelfGPUMul);
  bind.def("mmd_rbf_mul", &cpptools::rbf::MMDGPUMul);
}