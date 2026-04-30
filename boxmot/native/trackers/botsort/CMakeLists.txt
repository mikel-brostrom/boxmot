cmake_minimum_required(VERSION 3.16)
project(boxmot_botsort_cpp LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")
include(BoxMOTNative)

boxmot_add_native_tracker(
    NAME botsort
    OPENCV_COMPONENTS calib3d core dnn imgcodecs imgproc video
    CORE_SOURCES
        src/cmc.cpp
        src/data_io.cpp
        src/kalman_filter.cpp
        src/reid_onnx.cpp
        src/track.cpp
        src/tracker.cpp
)
