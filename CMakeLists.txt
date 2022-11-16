cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(RayTracer
    VERSION 1.0.0
    LANGUAGES CXX CUDA    
)

set(src_files
    ./src/main.cu
    ./src/Colour.cu
    # ./src/vec3.cpp
    # ./src/HittableList.cpp
    # ./src/Sphere.cpp
    # ./src/Camera.cpp
)

add_executable(${PROJECT_NAME}
    ${src_files}
)

# Enable CUDA support
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC include)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
