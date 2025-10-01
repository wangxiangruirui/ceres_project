cmake_minimum_required(VERSION 3.16)
project(ceres_project)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -DNDEBUG)
    add_link_options(-flto)
endif()

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 Ceres Solver
find_package(Ceres REQUIRED)

# 查找 Eigen3
find_package(Eigen3 REQUIRED)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 创建可执行文件
add_executable(ceres_project src/main.cpp)

# 链接库
target_link_libraries(ceres_project Ceres::ceres Eigen3::Eigen ${OpenCV_LIBS})

# 编译选项
target_compile_options(ceres_project PRIVATE -Wall -Wextra)
