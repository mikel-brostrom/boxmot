cmake_minimum_required(VERSION 3.16)
project(boxmot_botsort_cpp LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
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

# A small parity probe used by the Python unit suite to compare the native OBB
# camera-motion state transform with transform_obb_kalman_state. Keeping it out
# of the default build avoids adding test-only work to runtime native builds.
add_executable(botsort_obb_cmc_probe EXCLUDE_FROM_ALL tests/obb_cmc_probe.cpp)
target_compile_features(botsort_obb_cmc_probe PRIVATE cxx_std_17)
target_link_libraries(botsort_obb_cmc_probe PRIVATE botsort_core)
boxmot_enable_native_warnings(botsort_obb_cmc_probe)
