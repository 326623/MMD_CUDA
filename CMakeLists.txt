cmake_minimum_required(VERSION 3.13.4)
project(HPCorpIET)
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_COMPILER "clang++")
# set(CMAKE_CXX_FLAGS "-stdlib=libc++ -std=c++11")
if (EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
  include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
  conan_basic_setup(TARGETS)
else ()
  if (NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/conan.cmake")
    message(STATUS "conan.cmake not found")
    message(FATAL_ERROR "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
  else ()
    include(${CMAKE_SOURCE_DIR}/cmake/conan.cmake)
    conan_cmake_run(CONANFILE conanfile.txt
            BASIC_SETUP CMAKE_TARGETS
            BUILD missing)
  endif ()
endif ()

if (APPLE)
  set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_C_LIB_NAMES "omp")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.a")
endif ()
# cmake_policy(SET CMP0068 NEW)
#  -Rpass-analysis=loop-vectorize
# add_definitions("-std=c++11 -ffast-math -funsafe-math-optimizations")

# find_package(GTest REQUIRED)
# find_package(Boost REQUIRED COMPONENTS python)
# find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(CUDA)

if ($ENV{CLION_IDE})
include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/cpptools")
endif ()

# Cannot build this as shared lib without modifying to source code on Windows
# Luckily pybind11 seems to have done the job for us on Windows. Thanks.
add_library(cpu_kernels STATIC
        "${CMAKE_SOURCE_DIR}/cpptools/mmd_cpu.cc")
target_include_directories(cpu_kernels PRIVATE
        "${CMAKE_SOURCE_DIR}/cpptqdm"
        "${CMAKE_SOURCE_DIR}/cpptools")
target_link_libraries(cpu_kernels PUBLIC OpenMP::OpenMP_CXX)
set_target_properties(cpu_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)
# For GCC to enable vectorization
target_compile_options(cpu_kernels PRIVATE "-ffast-math"
        "-funsafe-math-optimizations" "-ftree-vectorize")

pybind11_add_module(cpptools "${CMAKE_SOURCE_DIR}/cpptools/pybind11_interface.cc")
target_include_directories(cpptools PRIVATE "${Python3_INCLUDE_DIRS}" "${CMAKE_SOURCE_DIR}/cpptools")
target_link_libraries(cpptools PRIVATE cpu_kernels CONAN_PKG::pybind11 ${Python3_LIBRARIES})

if (APPLE)
  target_include_directories(cpptools PRIVATE "/usr/local/opt/libomp/include")
endif ()

if (CUDA_FOUND)
  message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
  enable_language(CUDA)
  # most library I rely require std11 to proceed
  set(CMAKE_CUDA_STANDARD 11)
  # To enable Cuda Parsing for Clion
  #if ($ENV{CLION_IDE})
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  #endif ()
  set(CMAKE_CUDA_FLAGS "--expt-extended-lambda")

  # Can't make this shared.
  add_library(cuda_kernels STATIC
          "${CMAKE_SOURCE_DIR}/cpptools/mmd_cuda.cu"
          "${CMAKE_SOURCE_DIR}/cpptools/mmd_cuda_linear.cu"
          "${CMAKE_SOURCE_DIR}/cpptools/mmd_cuda_rbf.cu"
          "${CMAKE_SOURCE_DIR}/cpptools/utils_cuda.cc")
  target_include_directories(cuda_kernels PRIVATE
          "${CMAKE_SOURCE_DIR}/cpptools"
          ${CUDA_INCLUDE_DIRS})
  target_link_libraries(cuda_kernels PRIVATE ${CUDA_LIBRARIES})
  set_target_properties(cuda_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

  pybind11_add_module(cudatools
          "${CMAKE_SOURCE_DIR}/cpptools/pybind11_interface_cuda.cc")
  target_include_directories(cudatools PRIVATE
          "${Python3_INCLUDE_DIRS}"
          "${CMAKE_SOURCE_DIR}/cpptools")
  target_link_libraries(cudatools PRIVATE ${Python3_LIBRARIES} CONAN_PKG::pybind11 cuda_kernels)

  add_subdirectory("examples/benchmark")
  add_subdirectory("examples/unit_test")
endif ()
