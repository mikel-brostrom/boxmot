find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/BoxMOTNativeCompiler.cmake)

add_library(boxmot_tracker_base STATIC
    src/assignment.cpp
    src/io.cpp
)

target_include_directories(boxmot_tracker_base
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(boxmot_tracker_base
    PUBLIC
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

boxmot_enable_native_warnings(boxmot_tracker_base)
