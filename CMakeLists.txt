cmake_minimum_required(VERSION 3.0.0)
project(RayTracer VERSION 0.1.0)

include(CTest)
enable_testing()


set(src_files
    ./src/main.cpp
    ./src/Colour.cpp
    ./src/vec3.cpp
)

add_executable(${PROJECT_NAME}
    ${src_files}
)
target_include_directories(${PROJECT_NAME} PUBLIC ./include)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
