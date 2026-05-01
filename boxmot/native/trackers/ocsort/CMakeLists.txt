cmake_minimum_required(VERSION 3.16)
project(boxmot_ocsort_cpp LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")
include(BoxMOTNative)

boxmot_add_native_tracker(
    NAME ocsort
    CORE_SOURCES
        src/data_io.cpp
        src/kalman_filter.cpp
        src/tracker.cpp
)
