cmake_minimum_required(VERSION 3.26)

project(CUDAGalaxy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(SFML
        GIT_REPOSITORY https://github.com/SFML/SFML.git
        GIT_TAG 3.0.1
        GIT_SHALLOW ON
        EXCLUDE_FROM_ALL
        SYSTEM)
FetchContent_MakeAvailable(SFML)

add_executable(CUDAGalaxy
        src/main.cu
        src/kernel.cu
        src/helpers.cpp
)

target_link_libraries(CUDAGalaxy PRIVATE SFML::Graphics SFML::Window)

set_target_properties(CUDAGalaxy PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON )
