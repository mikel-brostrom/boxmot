cmake_minimum_required(VERSION 3.16)

project(boxmot_native_trackers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS calib3d core dnn imgcodecs imgproc video)
find_package(Eigen3 REQUIRED NO_MODULE)

add_subdirectory(base)
add_subdirectory(botsort)
add_subdirectory(bytetrack)
add_subdirectory(ocsort)
add_subdirectory(sfsort)
