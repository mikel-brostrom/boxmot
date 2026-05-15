cmake_minimum_required(VERSION 3.16)
project(boxmot_deepstream_adapter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Required dependencies
# ---------------------------------------------------------------------------

# CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# TensorRT
# Try to find TensorRT via environment or standard paths
if(NOT DEFINED TENSORRT_ROOT)
    if(EXISTS "/usr/lib/x86_64-linux-gnu/libnvinfer.so")
        set(TENSORRT_ROOT "/usr")
    elseif(EXISTS "/usr/local/tensorrt")
        set(TENSORRT_ROOT "/usr/local/tensorrt")
    elseif(DEFINED ENV{TENSORRT_ROOT})
        set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
    endif()
endif()

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS "${TENSORRT_ROOT}/include" "/usr/include/x86_64-linux-gnu"
)
find_library(TENSORRT_NVINFER
    NAMES nvinfer
    HINTS "${TENSORRT_ROOT}/lib" "/usr/lib/x86_64-linux-gnu"
)
find_library(TENSORRT_NVONNXPARSER
    NAMES nvonnxparser
    HINTS "${TENSORRT_ROOT}/lib" "/usr/lib/x86_64-linux-gnu"
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER OR NOT TENSORRT_NVONNXPARSER)
    message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=<path> or install TensorRT.")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${TENSORRT_NVINFER}")
message(STATUS "TensorRT nvonnxparser: ${TENSORRT_NVONNXPARSER}")

# OpenCV
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc dnn)

# Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

# DeepStream headers
if(NOT DEFINED DEEPSTREAM_ROOT)
    if(EXISTS "/opt/nvidia/deepstream/deepstream")
        set(DEEPSTREAM_ROOT "/opt/nvidia/deepstream/deepstream")
    elseif(DEFINED ENV{DEEPSTREAM_ROOT})
        set(DEEPSTREAM_ROOT "$ENV{DEEPSTREAM_ROOT}")
    endif()
endif()

find_path(DEEPSTREAM_INCLUDE_DIR
    NAMES nvdstracker.h
    HINTS
        "${DEEPSTREAM_ROOT}/sources/includes"
        "${DEEPSTREAM_ROOT}/include"
        "/opt/nvidia/deepstream/deepstream/sources/includes"
)

if(NOT DEEPSTREAM_INCLUDE_DIR)
    message(FATAL_ERROR
        "DeepStream headers not found. Set -DDEEPSTREAM_ROOT=<path>. "
        "Expected nvdstracker.h in <root>/sources/includes/")
endif()

message(STATUS "DeepStream headers: ${DEEPSTREAM_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# Optional: link against BoxMOT's existing native tracker static libraries
# ---------------------------------------------------------------------------
option(BOXMOT_DS_LINK_TRACKERS
    "Link against BoxMOT native tracker cores (botsort, bytetrack, etc.)" OFF)

if(BOXMOT_DS_LINK_TRACKERS)
    # Assume the parent trackers have been built
    if(NOT TARGET boxmot_tracker_base)
        get_filename_component(_base_dir "${CMAKE_CURRENT_LIST_DIR}/../base" ABSOLUTE)
        add_subdirectory("${_base_dir}" "${CMAKE_BINARY_DIR}/_boxmot_native_base")
    endif()
endif()

# ---------------------------------------------------------------------------
# Build the shared library: libnvds_boxmot_tracker.so
# ---------------------------------------------------------------------------
add_library(nvds_boxmot_tracker SHARED
    src/nvds_tracker_api.cpp
    src/adapter_context.cpp
    src/config_parser.cpp
    src/reid_tensorrt.cpp
)

target_compile_features(nvds_boxmot_tracker PUBLIC cxx_std_17)

target_include_directories(nvds_boxmot_tracker
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${DEEPSTREAM_INCLUDE_DIR}
        ${TENSORRT_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(nvds_boxmot_tracker
    PRIVATE
        ${TENSORRT_NVINFER}
        ${TENSORRT_NVONNXPARSER}
        CUDA::cudart
        ${OpenCV_LIBS}
        Eigen3::Eigen
)

if(BOXMOT_DS_LINK_TRACKERS AND TARGET boxmot_tracker_base)
    target_link_libraries(nvds_boxmot_tracker PRIVATE boxmot_tracker_base)
endif()

# Export symbols for the NvDsTracker C API
set_target_properties(nvds_boxmot_tracker PROPERTIES
    OUTPUT_NAME "nvds_boxmot_tracker"
    PREFIX "lib"
    # Only export the NvMOT_* symbols
    C_VISIBILITY_PRESET default
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Compiler warnings
target_compile_options(nvds_boxmot_tracker PRIVATE -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS nvds_boxmot_tracker
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/config_tracker_boxmot.yml
    DESTINATION etc
)

message(STATUS "BoxMOT DeepStream adapter configured successfully.")
message(STATUS "  Tracker algorithms: BoTSORT, ByteTrack, OCSORT, SFSORT, OccluBoost")
message(STATUS "  ReID backend: TensorRT (FP32/FP16/INT8)")
message(STATUS "  Output: libnvds_boxmot_tracker.so")
