cmake_minimum_required(VERSION 3.26)

project(CUDAGalaxy LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)


include(FetchContent)
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

add_library(glad_gl_core STATIC external/glad/src/glad.c)
target_include_directories(glad_gl_core PUBLIC ./external/glad/include)

add_executable(CUDAGalaxy
        src/main.cu
        src/kernel.cu
        src/helpers.cpp
        src/gl.cpp
)

target_link_libraries(CUDAGalaxy PRIVATE glfw glad_gl_core)

set_target_properties(CUDAGalaxy PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON )
