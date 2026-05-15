cmake_minimum_required(VERSION 3.16)

project(boxmot_native_trackers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS calib3d core dnn imgcodecs imgproc video)
find_package(Eigen3 REQUIRED NO_MODULE)

add_subdirectory(base)
add_subdirectory(botsort)
add_subdirectory(bytetrack)
add_subdirectory(occluboost)
add_subdirectory(ocsort)
add_subdirectory(sfsort)

# DeepStream adapter (optional — requires TensorRT + DeepStream SDK)
option(BOXMOT_BUILD_DEEPSTREAM "Build the DeepStream Gst-nvtracker adapter" OFF)
if(BOXMOT_BUILD_DEEPSTREAM)
    add_subdirectory(deepstream)
endif()

# ---------------------------------------------------------------------------
# Wheel install rules
# ---------------------------------------------------------------------------
# When configured by scikit-build-core (via the repo-root CMakeLists.txt) we
# install the C++ shared libraries (and replay executables) inside the wheel,
# alongside their respective Python wrapper modules. This makes
# ``import boxmot`` find pre-built ``*_capi.{so,dylib,dll}`` files without any
# runtime CMake invocation.
#
# Standalone configurations (per-tracker ``cmake -S boxmot/native/trackers/X``)
# leave install rules disabled by default to preserve the pre-existing
# editable-install build flow.
option(BOXMOT_INSTALL_NATIVE "Install native targets into the wheel layout" OFF)

if(BOXMOT_INSTALL_NATIVE)
    set(_BOXMOT_NATIVE_PKG_PREFIX "boxmot/native/trackers")

    # Per-tracker shared libraries and replay executables.
    foreach(_tracker IN ITEMS botsort bytetrack occluboost ocsort sfsort)
        set(_dest "${_BOXMOT_NATIVE_PKG_PREFIX}/${_tracker}")
        if(TARGET ${_tracker}_capi)
            install(TARGETS ${_tracker}_capi
                LIBRARY DESTINATION ${_dest}
                RUNTIME DESTINATION ${_dest}
                ARCHIVE DESTINATION ${_dest}
            )
        endif()
        if(TARGET ${_tracker}_replay)
            install(TARGETS ${_tracker}_replay
                RUNTIME DESTINATION ${_dest}
            )
        endif()
    endforeach()

    # Shared ReID C ABI lives next to ``boxmot/native/trackers/base/``.
    if(TARGET reid_capi)
        install(TARGETS reid_capi
            LIBRARY DESTINATION "${_BOXMOT_NATIVE_PKG_PREFIX}/base"
            RUNTIME DESTINATION "${_BOXMOT_NATIVE_PKG_PREFIX}/base"
            ARCHIVE DESTINATION "${_BOXMOT_NATIVE_PKG_PREFIX}/base"
        )
    endif()
endif()
