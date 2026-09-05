cmake_minimum_required(VERSION 3.16)

project(boxmot_native_trackers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The aggregate build publishes the standalone ReID C ABI unless explicitly
# disabled. Per-tracker builds never add the independent ReID subtree.
option(BOXMOT_BUILD_NATIVE_REID "Build the independent native ReID runtime and C ABI" ON)

add_subdirectory(trackers/base)
if(BOXMOT_BUILD_NATIVE_REID)
    add_subdirectory(reid)
endif()
add_subdirectory(trackers/botsort)
add_subdirectory(trackers/bytetrack)
add_subdirectory(trackers/occluboost)
add_subdirectory(trackers/ocsort)
add_subdirectory(trackers/sfsort)

# Focused native identity regression probe. It is excluded from production
# builds and compiled explicitly by the Python/native test suite.
add_executable(boxmot_native_identity_probe EXCLUDE_FROM_ALL tests/native_identity_probe.cpp)
target_compile_features(boxmot_native_identity_probe PRIVATE cxx_std_17)
target_link_libraries(boxmot_native_identity_probe PRIVATE
    botsort_core
    bytetrack_core
    occluboost_core
    ocsort_core
    sfsort_core
)
boxmot_enable_native_warnings(boxmot_native_identity_probe)

# ---------------------------------------------------------------------------
# Wheel install rules
# ---------------------------------------------------------------------------
# When configured by scikit-build-core (via the repo-root CMakeLists.txt) we
# install the C++ shared libraries inside the wheel,
# alongside their respective C++ source directories. This makes
# ``import boxmot`` find pre-built ``*_capi.{so,dylib,dll}`` files without any
# runtime CMake invocation.
#
# Standalone configurations (per-tracker ``cmake -S boxmot/native/cpp/trackers/X``)
# leave install rules disabled by default to preserve the pre-existing
# editable-install build flow.
option(BOXMOT_INSTALL_NATIVE "Install native targets into the wheel layout" OFF)

if(BOXMOT_INSTALL_NATIVE)
    set(_BOXMOT_NATIVE_PKG_PREFIX "boxmot/native/cpp/trackers")

    # Per-tracker typed v2 shared libraries.
    foreach(_tracker IN ITEMS botsort bytetrack occluboost ocsort sfsort)
        set(_dest "${_BOXMOT_NATIVE_PKG_PREFIX}/${_tracker}")
        if(TARGET ${_tracker}_capi)
            install(TARGETS ${_tracker}_capi
                LIBRARY DESTINATION ${_dest}
                RUNTIME DESTINATION ${_dest}
                ARCHIVE DESTINATION ${_dest}
            )
        endif()
    endforeach()

    # Standalone ReID C ABI is installed with its independent C++ sources.
    if(TARGET reid_capi)
        install(TARGETS reid_capi
            LIBRARY DESTINATION "boxmot/native/cpp/reid"
            RUNTIME DESTINATION "boxmot/native/cpp/reid"
            ARCHIVE DESTINATION "boxmot/native/cpp/reid"
        )
    endif()
endif()
