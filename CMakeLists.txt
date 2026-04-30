cmake_minimum_required(VERSION 3.16)

# Root CMake project consumed by ``scikit-build-core`` when building the
# ``boxmot`` wheel. Configures and installs the native (C++) tracker
# shared libraries directly into the wheel layout under
# ``boxmot/native/trackers/<name>/``.
#
# This file is intentionally thin: the real per-tracker logic lives in
# ``boxmot/native/trackers/CMakeLists.txt`` and per-tracker ``CMakeLists.txt``
# files. Standalone CMake configurations against those subprojects continue
# to work for editable / dev workflows.

project(boxmot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Always install native artifacts when the project is built from this root.
# scikit-build-core invokes ``cmake --install`` against the wheel tree.
set(BOXMOT_INSTALL_NATIVE ON CACHE BOOL "Install native targets into the wheel" FORCE)

# Allow consumers to skip native compilation entirely (Python-only wheel).
option(BOXMOT_BUILD_NATIVE "Compile bundled C++ tracker shared libraries" ON)

# By default, missing C++ dependencies (OpenCV, Eigen3, a working compiler)
# do **not** fail ``pip install boxmot``. They simply skip the native build,
# leaving the user with the pure-Python trackers and the option to compile
# the C++ libraries later via ``boxmot build``.
#
# Wheel builders (``cibuildwheel`` / release CI) should set
# ``-DBOXMOT_REQUIRE_NATIVE=ON`` so any missing dependency fails the build
# loudly instead of silently shipping a Python-only wheel.
option(BOXMOT_REQUIRE_NATIVE "Fail the build if native dependencies are missing" OFF)

# ---------------------------------------------------------------------------
# Soft probe for native build prerequisites
# ---------------------------------------------------------------------------
set(_BOXMOT_NATIVE_OK ON)
set(_BOXMOT_NATIVE_MISSING "")

if(BOXMOT_BUILD_NATIVE)
    find_package(OpenCV 4 QUIET COMPONENTS calib3d core dnn imgcodecs imgproc video)
    if(NOT OpenCV_FOUND)
        set(_BOXMOT_NATIVE_OK OFF)
        list(APPEND _BOXMOT_NATIVE_MISSING "OpenCV 4 (with calib3d, core, dnn, imgcodecs, imgproc, video)")
    endif()

    find_package(Eigen3 QUIET NO_MODULE)
    if(NOT Eigen3_FOUND)
        set(_BOXMOT_NATIVE_OK OFF)
        list(APPEND _BOXMOT_NATIVE_MISSING "Eigen3 (>= 3.3, with CMake config)")
    endif()
endif()

if(BOXMOT_BUILD_NATIVE AND NOT _BOXMOT_NATIVE_OK)
    string(REPLACE ";" ", " _missing_str "${_BOXMOT_NATIVE_MISSING}")
    if(BOXMOT_REQUIRE_NATIVE)
        message(FATAL_ERROR
            "boxmot: BOXMOT_REQUIRE_NATIVE=ON but the following native build "
            "dependencies were not found: ${_missing_str}. "
            "Install them (e.g. 'apt-get install libopencv-dev libeigen3-dev', "
            "'brew install opencv eigen', or 'vcpkg install opencv4 eigen3') "
            "and retry, or pass -DBOXMOT_BUILD_NATIVE=OFF to skip native "
            "compilation."
        )
    endif()
    message(WARNING
        "boxmot: skipping native (C++) tracker compilation. "
        "Missing dependencies: ${_missing_str}. "
        "Pure-Python trackers will still work. To enable the C++ backends "
        "later, install the missing dependencies and run 'boxmot build', or "
        "reinstall with the dependencies present."
    )
endif()

if(BOXMOT_BUILD_NATIVE AND _BOXMOT_NATIVE_OK)
    add_subdirectory(boxmot/native/trackers)
elseif(NOT BOXMOT_BUILD_NATIVE)
    message(STATUS "boxmot: skipping native tracker compilation (BOXMOT_BUILD_NATIVE=OFF)")
endif()

