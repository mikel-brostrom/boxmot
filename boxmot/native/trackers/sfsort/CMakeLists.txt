cmake_minimum_required(VERSION 3.16)

project(boxmot_sfsort_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(Eigen3 REQUIRED NO_MODULE)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/BoxMOTNativeCompiler.cmake)

if(NOT TARGET boxmot_tracker_base)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../base ${CMAKE_CURRENT_BINARY_DIR}/base)
endif()

add_library(sfsort_core
    src/data_io.cpp
    src/tracker.cpp
)

set_target_properties(sfsort_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(sfsort_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(sfsort_core
    PUBLIC
        Eigen3::Eigen
        boxmot_tracker_base
        ${OpenCV_LIBS}
)

boxmot_enable_native_warnings(sfsort_core)

add_library(sfsort_capi SHARED src/c_api.cpp)
target_include_directories(sfsort_capi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sfsort_capi PRIVATE sfsort_core)
target_compile_definitions(sfsort_capi PRIVATE BOXMOT_SFSORT_BUILDING_DLL)
set_target_properties(sfsort_capi PROPERTIES OUTPUT_NAME sfsort_capi PREFIX "")

boxmot_enable_native_warnings(sfsort_capi)

add_executable(sfsort_replay src/main.cpp)
target_link_libraries(sfsort_replay PRIVATE sfsort_core)
