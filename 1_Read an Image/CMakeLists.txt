cmake_minimum_required (VERSION 3.17)
cmake_policy(SET CMP0012 NEW)
project (ReadImage)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# OpenCV package
find_package(OpenCV REQUIRED)
message("[INFO] OpenCV version ${OpenCV_VERSION}")

add_executable(${PROJECT_NAME} main.cpp)
#include_directories(${OpenCV_INCLUDE_DIRS})
#link_directories(${OpenCV_LIB_DIR})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})