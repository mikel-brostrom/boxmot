# BoxMOT native — shared CMake helpers.
#
# Goal: every per-tracker CMakeLists.txt collapses down to a single
# ``boxmot_add_native_tracker(...)`` call. Project-wide concerns
# (warnings, C++ standard, optional ``-Werror``) live here so they can be
# changed in one place.

include_guard(GLOBAL)

get_filename_component(_BOXMOT_NATIVE_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(_BOXMOT_NATIVE_ROOT "${_BOXMOT_NATIVE_CMAKE_DIR}/.." ABSOLUTE)

# ---------------------------------------------------------------------------
# Project-wide defaults.
# ---------------------------------------------------------------------------
option(BOXMOT_NATIVE_WERROR "Treat native compiler warnings as errors" OFF)

# ---------------------------------------------------------------------------
# Apply the BoxMOT-wide warning set to ``target``.
#
# Always: ``-Wall -Wextra -Wpedantic`` (or ``/W4`` on MSVC).
# When ``BOXMOT_NATIVE_WERROR=ON``: also promote them to errors.
# ---------------------------------------------------------------------------
function(boxmot_enable_native_warnings target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "boxmot_enable_native_warnings: '${target}' is not a target")
    endif()

    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
        if(BOXMOT_NATIVE_WERROR)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(BOXMOT_NATIVE_WERROR)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Resolve OpenCV / Eigen / boxmot_tracker_base once per directory.
# ---------------------------------------------------------------------------
function(boxmot_require_native_deps)
    set(options "")
    set(one_value "")
    set(multi_value OPENCV_COMPONENTS)
    cmake_parse_arguments(BX "${options}" "${one_value}" "${multi_value}" ${ARGN})

    if(NOT BX_OPENCV_COMPONENTS)
        set(BX_OPENCV_COMPONENTS core imgcodecs imgproc)
    endif()

    find_package(OpenCV 4 REQUIRED COMPONENTS ${BX_OPENCV_COMPONENTS})
    find_package(Eigen3 REQUIRED NO_MODULE)

    if(NOT TARGET boxmot_tracker_base)
        # Pull the shared base library in if the per-tracker project is
        # being built standalone (e.g. ``cmake -S boxmot/native/cpp/trackers/X``).
        get_filename_component(_boxmot_native_dir
            "${_BOXMOT_NATIVE_ROOT}/trackers/base" ABSOLUTE)
        add_subdirectory("${_boxmot_native_dir}"
            "${CMAKE_BINARY_DIR}/_boxmot_native_base")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# Add a complete native tracker package: <name>_core (static), <name>_capi
# (shared C ABI), <name>_replay (executable).
#
# Usage:
#   boxmot_add_native_tracker(
#       NAME       botsort
#       CORE_SOURCES
#           src/cmc.cpp
#           src/data_io.cpp
#           src/kalman_filter.cpp
#           src/reid_onnx.cpp
#           src/track.cpp
#           src/tracker.cpp
#       OPENCV_COMPONENTS calib3d core dnn imgcodecs imgproc video
#   )
#
# Optional:
#   CAPI_SOURCES <files...>     (defaults to src/c_api.cpp)
#   REPLAY_SOURCES <files...>   (defaults to src/main.cpp; pass NONE to skip)
#   EXTRA_PUBLIC_LIBS <libs...> (extra interface libs for <name>_core)
# ---------------------------------------------------------------------------
function(boxmot_add_native_tracker)
    set(options "")
    set(one_value NAME)
    set(multi_value
        CORE_SOURCES
        CAPI_SOURCES
        REPLAY_SOURCES
        OPENCV_COMPONENTS
        EXTRA_PUBLIC_LIBS
    )
    cmake_parse_arguments(BX "${options}" "${one_value}" "${multi_value}" ${ARGN})

    if(NOT BX_NAME)
        message(FATAL_ERROR "boxmot_add_native_tracker: NAME is required")
    endif()
    if(NOT BX_CORE_SOURCES)
        message(FATAL_ERROR "boxmot_add_native_tracker(${BX_NAME}): CORE_SOURCES is required")
    endif()
    if(NOT BX_CAPI_SOURCES)
        set(BX_CAPI_SOURCES src/c_api.cpp)
    endif()
    if(NOT BX_REPLAY_SOURCES)
        set(BX_REPLAY_SOURCES src/main.cpp)
    endif()

    boxmot_require_native_deps(OPENCV_COMPONENTS ${BX_OPENCV_COMPONENTS})

    set(_core "${BX_NAME}_core")
    set(_capi "${BX_NAME}_capi")
    set(_replay "${BX_NAME}_replay")

    # ---- core static library ---------------------------------------------
    add_library(${_core} STATIC ${BX_CORE_SOURCES})
    target_compile_features(${_core} PUBLIC cxx_std_17)
    set_target_properties(${_core} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${_core}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(${_core}
        PUBLIC
            Eigen3::Eigen
            boxmot_tracker_base
            ${OpenCV_LIBS}
            ${BX_EXTRA_PUBLIC_LIBS}
    )
    boxmot_enable_native_warnings(${_core})

    # ---- C ABI shared library --------------------------------------------
    string(TOUPPER "${BX_NAME}" _name_upper)
    add_library(${_capi} SHARED ${BX_CAPI_SOURCES})
    target_compile_features(${_capi} PUBLIC cxx_std_17)
    target_link_libraries(${_capi} PRIVATE ${_core})
    target_compile_definitions(${_capi}
        PRIVATE BOXMOT_${_name_upper}_BUILDING_DLL)
    set_target_properties(${_capi} PROPERTIES
        OUTPUT_NAME ${_capi}
        PREFIX ""
        # Non-API symbols stay private to the shared lib.
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    boxmot_enable_native_warnings(${_capi})

    # ---- replay executable -----------------------------------------------
    if(NOT BX_REPLAY_SOURCES STREQUAL "NONE")
        add_executable(${_replay} ${BX_REPLAY_SOURCES})
        target_compile_features(${_replay} PRIVATE cxx_std_17)
        target_link_libraries(${_replay} PRIVATE ${_core})
        boxmot_enable_native_warnings(${_replay})
    endif()
endfunction()
