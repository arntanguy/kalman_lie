cmake_minimum_required(VERSION 2.8)
project(KalmanLie CXX)

# TODO: Use VERSION argument to project command after raising the minimum
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})

option(BUILD_LIE_EXAMPLE "Build simple Lie example." ON) 
option(BUILD_ROS_LIE_EXAMPLE "Build ROS Lie example." ON) 

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

if(BUILD_LIE_EXAMPLE)
add_executable(example_lie "examples/Lie/main.cpp")
endif()

if(BUILD_ROS_LIE_EXAMPLE)
add_subdirectory("examples/lie_ros")
endif()
