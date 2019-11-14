project(UNIT_TESTING)

add_executable(floating_point_error
        "${CMAKE_SOURCE_DIR}/examples/unit_test/floating_point_error.cc" floating_point_error_mul.cc)
target_include_directories(floating_point_error PRIVATE "${CMAKE_SOURCE_DIR}/cpptools")
target_link_libraries(floating_point_error PRIVATE
        cuda_kernels cpu_kernels CONAN_PKG::gtest)
