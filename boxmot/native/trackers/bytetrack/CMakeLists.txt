cmake_minimum_required(VERSION 3.16)

project(boxmot_bytetrack_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(Eigen3 REQUIRED NO_MODULE)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/BoxMOTNativeCompiler.cmake)

if(NOT TARGET boxmot_tracker_base)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../base ${CMAKE_CURRENT_BINARY_DIR}/base)
endif()

add_library(bytetrack_core
    src/data_io.cpp
    src/kalman_filter.cpp
    src/track.cpp
    src/tracker.cpp
)

set_target_properties(bytetrack_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(bytetrack_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(bytetrack_core
    PUBLIC
        Eigen3::Eigen
    boxmot_tracker_base
        ${OpenCV_LIBS}
)

boxmot_enable_native_warnings(bytetrack_core)

add_library(bytetrack_capi SHARED src/c_api.cpp)
target_include_directories(bytetrack_capi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(bytetrack_capi PRIVATE bytetrack_core)
target_compile_definitions(bytetrack_capi PRIVATE BOXMOT_BYTETRACK_BUILDING_DLL)
set_target_properties(bytetrack_capi PROPERTIES OUTPUT_NAME bytetrack_capi PREFIX "")

boxmot_enable_native_warnings(bytetrack_capi)

add_executable(bytetrack_replay src/main.cpp)
target_link_libraries(bytetrack_replay PRIVATE bytetrack_core)
