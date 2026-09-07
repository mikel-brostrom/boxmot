# Native BotSort

This directory contains the C++17 BotSort core and typed v2 shared library.

## Requirements

- C++17 compiler: GCC 9+, Clang 10+, or MSVC 2019+
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

## Build

```bash
cmake -S boxmot/native/cpp/trackers/botsort -B build/native/botsort -DCMAKE_BUILD_TYPE=Release
cmake --build build/native/botsort --config Release --target botsort_capi
```

If CMake cannot locate OpenCV or Eigen3 automatically, pass `-DOpenCV_DIR=...` and/or `-DEigen3_DIR=...` to the configure command.

## Role In BoxMOT

Python supplies canonical detections and generated or precomputed embedding
buffers through the live C ABI. The C++ tracker owns no model, download, or
cache logic.

Detection/layout support:

- Inputs use separate typed buffers: AABB geometry has four float columns and
  OBB geometry has five.
- Scores and embeddings are float buffers; class and detection IDs are
  `int64` buffers.
- The library allocates typed results, which callers release explicitly.
