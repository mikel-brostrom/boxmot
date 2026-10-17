cmake_minimum_required(VERSION 3.16)
project(boxmot_occluboost_cpp LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(BoxMOTNative)

boxmot_add_native_tracker(
    NAME occluboost
    OPENCV_COMPONENTS calib3d core imgproc video
    CORE_SOURCES
        src/association.cpp
        src/cmc.cpp
        src/kalman_filter.cpp
        src/track.cpp
        src/tracker.cpp
)

add_executable(occluboost_cmc_mask_probe EXCLUDE_FROM_ALL tests/cmc_mask_probe.cpp)
target_compile_features(occluboost_cmc_mask_probe PRIVATE cxx_std_17)
target_link_libraries(occluboost_cmc_mask_probe PRIVATE occluboost_core)
boxmot_enable_native_warnings(occluboost_cmc_mask_probe)

add_executable(occluboost_cmc_state_probe EXCLUDE_FROM_ALL tests/cmc_state_probe.cpp)
target_compile_features(occluboost_cmc_state_probe PRIVATE cxx_std_17)
target_link_libraries(occluboost_cmc_state_probe PRIVATE occluboost_core)
boxmot_enable_native_warnings(occluboost_cmc_state_probe)

# Exercise the bounded CMC estimators and resolution-change state resets.
add_executable(occluboost_cmc_estimator_probe EXCLUDE_FROM_ALL ../../tests/cmc_estimator_probe.cpp)
target_compile_features(occluboost_cmc_estimator_probe PRIVATE cxx_std_17)
target_compile_definitions(occluboost_cmc_estimator_probe PRIVATE
    BOXMOT_CMC_HEADER="occluboost/cmc.hpp"
    BOXMOT_CMC_NAMESPACE=occluboost
)
target_link_libraries(occluboost_cmc_estimator_probe PRIVATE occluboost_core)
boxmot_enable_native_warnings(occluboost_cmc_estimator_probe)
