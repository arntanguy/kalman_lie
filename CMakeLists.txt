cmake_minimum_required(VERSION 2.8)
project(KalmanLie CXX)

# TODO: Use VERSION argument to project command after raising the minimum
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/include/kalman/cmake")
find_package(Eigen3 REQUIRED)

include_directories(
    "include"
    "include/kalman/include"
    "include/Sophus"
    ${Eigen3_INCLUDE_DIR}
)

set(KALMAN_CXX_FLAGS "-std=c++11")
# Flags
set(CMAKE_CXX_FLAGS "${KALMAN_CXX_FLAGS}") # -Wall -pedantic -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DEIGEN_NO_DEBUG")

add_executable(example_lie "examples/Lie/main.cpp")
