# Native BoTSORT

This directory contains the native C++17 BoTSORT implementation used by BoxMOT for:

- the standalone replay executable used by cached benchmark workflows
- the shared library used by live `track --tracker-backend cpp`

## Requirements

- C++17 compiler: GCC 9+, Clang 10+, or MSVC 2019+
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

## Build

```bash
cmake -S boxmot/native/cpp/trackers/botsort -B build/native/botsort -DCMAKE_BUILD_TYPE=Release
cmake --build build/native/botsort --config Release --target botsort_replay
```

If CMake cannot locate OpenCV or Eigen3 automatically, pass `-DOpenCV_DIR=...` and/or `-DEigen3_DIR=...` to the configure command.

## Role In BoxMOT

The native runner is designed for the cached replay stage used by `boxmot eval`, `tune`, and similar workflows:

1. Python generates detections into `runs/dets_n_embs/...`
2. `botsort_replay` consumes cached detections and writes MOT result files
3. Python runs the in-repo MOT metrics on the generated results

ReID support:

- Live native tracking can run ONNX ReID inference internally when the tracker is configured with an `.onnx` ReID model such as `models/lmbn_n_duke.onnx`.
- Native replay can fall back to ONNX ReID inference when an embedding cache is unavailable and an `.onnx` ReID model path is provided.
- Existing embedding caches are still reused when present.

Detection/layout support:

- Live native tracking accepts AABB detections with 6 columns and OBB detections with 7 columns.
- Cached replay accepts AABB caches with 7 columns and OBB caches with 8 columns.
- Native OBB replay writes MMOT-style corner outputs for the downstream evaluation flow.
