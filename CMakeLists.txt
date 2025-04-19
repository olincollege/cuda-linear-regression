cmake_minimum_required(VERSION 3.22)
project(CudaLinearRegression LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 52;61;75;86;89)
find_package(CUDAToolkit REQUIRED) 
include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})

set(CMAKE_CXX_CLANG_TIDY "clang-tidy" )

add_subdirectory(src)

find_library(CRITERION criterion REQUIRED)
set(CRITERION_FLAGS --verbose -j1)
add_subdirectory(test)
