cmake_minimum_required(VERSION 3.22)
project(CudaLinearRegression LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust to your GPU
set(CMAKE_CXX_CLANG_TIDY "clang-tidy" )

set(SOURCES
    main.cpp
)
add_subdirectory(src)
add_subdirectory(test)
find_library(CRITERION criterion REQUIRED)
set(CRITERION_FLAGS --verbose -j1)
