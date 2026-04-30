# Allow this directory to be configured standalone (e.g. by the
# ``boxmot.native.reid_capi`` Python loader) as well as via the parent
# ``boxmot/native/trackers`` project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(boxmot_tracker_base LANGUAGES CXX)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core dnn imgproc)

# When this CMakeLists.txt is configured as a standalone build (e.g. by the
# ``boxmot.native.reid_capi`` Python loader), the parent project that normally
# sets the C++ standard is absent. Default to C++17 so headers using
# ``std::filesystem`` / ``std::string_view`` continue to compile in that mode.
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# Optional ONNX Runtime backend for the shared ReID inference path.
# Resolution order:
#   1. ``-DBOXMOT_REID_ONNXRUNTIME=OFF`` disables it entirely.
#   2. ``-DONNXRUNTIME_ROOT=<dir>`` points to a manual install.
#   3. ``find_package(onnxruntime CONFIG)`` (works with newer brew/conda
#      installs that ship a CMake config).
#   4. Homebrew default ``/opt/homebrew/opt/onnxruntime`` on macOS.
option(BOXMOT_REID_ONNXRUNTIME "Build with ONNX Runtime backend for native ReID" ON)

set(BOXMOT_ORT_AVAILABLE OFF)
set(BOXMOT_ORT_TARGET "")

if(BOXMOT_REID_ONNXRUNTIME)
    if(NOT DEFINED ONNXRUNTIME_ROOT)
        if(EXISTS "/opt/homebrew/opt/onnxruntime")
            set(ONNXRUNTIME_ROOT "/opt/homebrew/opt/onnxruntime")
        elseif(EXISTS "/usr/local/opt/onnxruntime")
            set(ONNXRUNTIME_ROOT "/usr/local/opt/onnxruntime")
        endif()
    endif()

    find_package(onnxruntime CONFIG QUIET)
    if(onnxruntime_FOUND)
        set(BOXMOT_ORT_AVAILABLE ON)
        set(BOXMOT_ORT_TARGET onnxruntime::onnxruntime)
    elseif(ONNXRUNTIME_ROOT)
        find_path(ONNXRUNTIME_INCLUDE_DIR
            NAMES onnxruntime_cxx_api.h
            HINTS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/include/onnxruntime"
        )
        find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime
            HINTS "${ONNXRUNTIME_ROOT}/lib"
        )
        if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
            add_library(boxmot_onnxruntime UNKNOWN IMPORTED)
            set_target_properties(boxmot_onnxruntime PROPERTIES
                IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
            )
            set(BOXMOT_ORT_AVAILABLE ON)
            set(BOXMOT_ORT_TARGET boxmot_onnxruntime)
        endif()
    endif()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/BoxMOTNative.cmake)

add_library(boxmot_tracker_base STATIC
    src/assignment.cpp
    src/io.cpp
    src/reid_onnx.cpp
)

target_compile_features(boxmot_tracker_base PUBLIC cxx_std_17)
set_target_properties(boxmot_tracker_base PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(boxmot_tracker_base
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(boxmot_tracker_base
    PUBLIC
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

if(BOXMOT_ORT_AVAILABLE)
    target_compile_definitions(boxmot_tracker_base PUBLIC BOXMOT_HAS_ONNXRUNTIME)
    target_link_libraries(boxmot_tracker_base PUBLIC ${BOXMOT_ORT_TARGET})
    message(STATUS "boxmot: ReID backend includes ONNX Runtime (${BOXMOT_ORT_TARGET})")
else()
    message(STATUS "boxmot: ReID backend uses OpenCV DNN only (ONNX Runtime not found)")
endif()

boxmot_enable_native_warnings(boxmot_tracker_base)

# Standalone C ABI shared library for the ReID model. Used by the Python eval
# pipeline to populate the embedding cache via the same native ONNX path that
# the C++ trackers use at replay time.
add_library(reid_capi SHARED src/reid_capi.cpp)
target_compile_features(reid_capi PUBLIC cxx_std_17)
target_link_libraries(reid_capi PRIVATE boxmot_tracker_base)
target_compile_definitions(reid_capi PRIVATE BOXMOT_REID_CAPI_BUILDING_DLL)
set_target_properties(reid_capi PROPERTIES
    OUTPUT_NAME reid_capi
    PREFIX ""
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

boxmot_enable_native_warnings(reid_capi)
