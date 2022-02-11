cmake_minimum_required(VERSION 3.0)
project(NumberPlateRecognition)
set(CMAKE_CXX_STANDARD 11)

# Requires OpenCV
find_package(OpenCV 4.0.0 REQUIRED core highgui ml flann dnnn)
message(STATUS "OpenCV version : ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

set(SRC main.cpp DetectedRegions.cpp OCR.cpp Plate.cpp)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})