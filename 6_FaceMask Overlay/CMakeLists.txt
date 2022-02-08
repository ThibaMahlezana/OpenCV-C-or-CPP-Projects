cmake_minimum_required (VERSION 3.0) 
PROJECT(FaceMaskOverlay)
set(CMAKE_CXX_STANDARD 11)

# Requires OpenCV v4.0 or later
FIND_PACKAGE( OpenCV 4 REQUIRED )

ADD_EXECUTABLE( ${PROJECT_NAME} main.cpp )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${OPENCV_LIBS} -lprotobuf)
