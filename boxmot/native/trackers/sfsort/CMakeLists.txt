cmake_minimum_required(VERSION 3.16)
project(boxmot_sfsort_cpp LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")
include(BoxMOTNative)

boxmot_add_native_tracker(
    NAME sfsort
    CORE_SOURCES
        src/data_io.cpp
        src/tracker.cpp
)
