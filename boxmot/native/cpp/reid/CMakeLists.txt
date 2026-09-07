# Allow the native ReID runtime to be built either independently or through
# the aggregate ``boxmot/native/cpp`` project. Tracker projects never add this
# directory and therefore never discover or link a model runtime.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(boxmot_native_reid LANGUAGES CXX)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core dnn imgproc)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/BoxMOTNative.cmake)

# Optional ONNX Runtime backend for the standalone appearance encoder.
# Resolution order:
#   1. ``-DBOXMOT_REID_ONNXRUNTIME=OFF`` disables it entirely.
#   2. ``-DONNXRUNTIME_ROOT=<dir>`` points to a manual install.
#   3. ``find_package(onnxruntime CONFIG)`` uses a packaged CMake config.
#   4. Homebrew defaults are considered on macOS.
option(BOXMOT_REID_ONNXRUNTIME "Build native ReID with ONNX Runtime" ON)

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

add_library(boxmot_native_reid STATIC
    src/reid_inference_backend.cpp
    src/reid_onnx.cpp
)
target_compile_features(boxmot_native_reid PUBLIC cxx_std_17)
set_target_properties(boxmot_native_reid PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(boxmot_native_reid
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../include
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(boxmot_native_reid
    PUBLIC
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

if(BOXMOT_ORT_AVAILABLE)
    target_compile_definitions(boxmot_native_reid PRIVATE BOXMOT_HAS_ONNXRUNTIME)
    target_link_libraries(boxmot_native_reid PUBLIC ${BOXMOT_ORT_TARGET})
    message(STATUS "boxmot: ReID backend includes ONNX Runtime (${BOXMOT_ORT_TARGET})")
else()
    message(STATUS "boxmot: ReID backend uses OpenCV DNN only (ONNX Runtime not found)")
endif()

boxmot_enable_native_warnings(boxmot_native_reid)

add_library(reid_capi SHARED src/reid_capi.cpp)
target_compile_features(reid_capi PUBLIC cxx_std_17)
target_link_libraries(reid_capi PRIVATE boxmot_native_reid)
target_compile_definitions(reid_capi PRIVATE BOXMOT_REID_CAPI_BUILDING_DLL)
set_target_properties(reid_capi PROPERTIES
    OUTPUT_NAME reid_capi
    PREFIX ""
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

boxmot_enable_native_warnings(reid_capi)
