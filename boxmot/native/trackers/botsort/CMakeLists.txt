cmake_minimum_required(VERSION 3.16)

project(boxmot_botsort_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS calib3d core dnn imgcodecs imgproc video)
find_package(Eigen3 REQUIRED NO_MODULE)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/BoxMOTNativeCompiler.cmake)

if(NOT TARGET boxmot_tracker_base)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../base ${CMAKE_CURRENT_BINARY_DIR}/base)
endif()

add_library(botsort_core
    src/cmc.cpp
    src/data_io.cpp
    src/kalman_filter.cpp
    src/reid_onnx.cpp
    src/track.cpp
    src/tracker.cpp
)

set_target_properties(botsort_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(botsort_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(botsort_core
    PUBLIC
        Eigen3::Eigen
    boxmot_tracker_base
        ${OpenCV_LIBS}
)

boxmot_enable_native_warnings(botsort_core)

add_library(botsort_capi SHARED src/c_api.cpp)
target_include_directories(botsort_capi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(botsort_capi PRIVATE botsort_core)
target_compile_definitions(botsort_capi PRIVATE BOXMOT_BOTSORT_BUILDING_DLL)
set_target_properties(botsort_capi PROPERTIES OUTPUT_NAME botsort_capi PREFIX "")

boxmot_enable_native_warnings(botsort_capi)

add_executable(botsort_replay src/main.cpp)
target_link_libraries(botsort_replay PRIVATE botsort_core)
