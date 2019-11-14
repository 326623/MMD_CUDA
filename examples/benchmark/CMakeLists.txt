project(BENCHMARKING)

add_executable(mmd_benchmark_cpu "${CMAKE_SOURCE_DIR}/examples/benchmark/mmd_benchmark_cpu.cc")
target_include_directories(mmd_benchmark_cpu PRIVATE "${CMAKE_SOURCE_DIR}/cpptools")
target_link_libraries(mmd_benchmark_cpu PRIVATE cpu_kernels CONAN_PKG::benchmark)

if (CUDA_FOUND)
add_executable(mmd_benchmark_cuda
        "${CMAKE_SOURCE_DIR}/examples/benchmark/mmd_benchmark_cuda.cc" mmd_benchmark_cuda_mul.cc)
target_include_directories(mmd_benchmark_cuda PRIVATE "${CMAKE_SOURCE_DIR}/cpptools")
target_link_libraries(mmd_benchmark_cuda PRIVATE cuda_kernels CONAN_PKG::benchmark)
endif()