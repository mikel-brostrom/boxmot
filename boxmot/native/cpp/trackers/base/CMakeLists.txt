# Allow the model-free tracker base and its regression probes to be configured
# standalone as well as through a tracker or aggregate native project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(boxmot_tracker_base LANGUAGES CXX)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)

# A standalone build has no parent project to select the language standard.
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/BoxMOTNative.cmake)

# Association and assignment are the complete shared tracker base. Keep model
# sources and inference dependencies out of this target: every tracker core
# links it, including standalone per-tracker builds.
add_library(boxmot_tracker_base STATIC
    src/association.cpp
    src/assignment.cpp
)

target_compile_features(boxmot_tracker_base PUBLIC cxx_std_17)
set_target_properties(boxmot_tracker_base PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(boxmot_tracker_base
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/../../include
)

target_link_libraries(boxmot_tracker_base
    PUBLIC
        Eigen3::Eigen
)

boxmot_enable_native_warnings(boxmot_tracker_base)

# Small solver regression probe used by the Python native unit suite. Keep it
# outside the default build so production artifacts do not gain test-only work.
add_executable(boxmot_assignment_probe EXCLUDE_FROM_ALL tests/assignment_probe.cpp)
target_compile_features(boxmot_assignment_probe PRIVATE cxx_std_17)
target_link_libraries(boxmot_assignment_probe PRIVATE boxmot_tracker_base)
boxmot_enable_native_warnings(boxmot_assignment_probe)

add_executable(boxmot_association_probe EXCLUDE_FROM_ALL tests/association_probe.cpp)
target_compile_features(boxmot_association_probe PRIVATE cxx_std_17)
target_link_libraries(boxmot_association_probe PRIVATE boxmot_tracker_base)
boxmot_enable_native_warnings(boxmot_association_probe)
