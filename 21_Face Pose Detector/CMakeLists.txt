cmake_minimum_required(VERSION 3.0)
project(FacePoseDetector)
set(CMAKE_CXX_STANDARD 14)

# OpenCV library
set(OpenCV_DIR "" CACHE PATH "OpenCV 4 build Directory")
find_package(OpenCV 4 REQUIRED core face highgui calib3d)

# Boost Library
find_package(Boost REQUIRED system)
include_directories(${Boost_INCLUDE_DIR})

add_executable(${PROJECT} main.cpp)
target_link_libraries(${PROJECT} ${OpenCV_LIBRARIES})