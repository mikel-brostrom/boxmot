cmake_minimum_required(VERSION 3.16)

project(boxmot_occluboost_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS calib3d core dnn imgcodecs imgproc video)
find_package(Eigen3 REQUIRED NO_MODULE)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/BoxMOTNativeCompiler.cmake)

if(NOT TARGET boxmot_tracker_base)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../base ${CMAKE_CURRENT_BINARY_DIR}/base)
endif()

add_library(occluboost_core
    src/association.cpp
    src/cmc.cpp
    src/data_io.cpp
    src/kalman_filter.cpp
    src/reid_onnx.cpp
    src/track.cpp
    src/tracker.cpp
)

set_target_properties(occluboost_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(occluboost_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(occluboost_core
    PUBLIC
        Eigen3::Eigen
    boxmot_tracker_base
        ${OpenCV_LIBS}
)

boxmot_enable_native_warnings(occluboost_core)

add_library(occluboost_capi SHARED src/c_api.cpp)
target_include_directories(occluboost_capi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(occluboost_capi PRIVATE occluboost_core)
target_compile_definitions(occluboost_capi PRIVATE BOXMOT_OCCLUBOOST_BUILDING_DLL)
set_target_properties(occluboost_capi PROPERTIES OUTPUT_NAME occluboost_capi PREFIX "")

boxmot_enable_native_warnings(occluboost_capi)

add_executable(occluboost_replay src/main.cpp)
target_link_libraries(occluboost_replay PRIVATE occluboost_core)
