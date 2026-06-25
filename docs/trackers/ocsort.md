# OcSort

[Paper: Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking](https://arxiv.org/abs/2203.14360)

OC-SORT focuses on a specific failure mode in Kalman-filter trackers: error accumulation during occlusion and non-linear motion. The paper replaces a purely prediction-centric view with an observation-centric one, using detector observations to reconstruct a more reliable virtual trajectory across missed frames. That makes the tracker much more robust than vanilla SORT in crowded scenes while keeping the same simple online structure.

## What BoxMOT Needs For OcSort

- Detector only. ReID is not required.
- Supports both AABB and OBB detections in BoxMOT.
- A strong choice when you want a fast motion-only tracker but expect more non-linear motion or occlusion than ByteTrack handles comfortably.

## Native C++ Backend

BoxMOT ships a native C++17 OCSORT implementation under `boxmot/native/cpp/trackers/ocsort/`. It supports:

- cached replay for `eval`, `tune`, and `research`
- live `track` through `--tracker-backend cpp`
- both AABB and OBB detection layouts in the native tracker path

Requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot eval --benchmark mot17 --split ablation --tracker ocsort --tracker-backend cpp
boxmot track --tracker ocsort --tracker-backend cpp --source 0
```

The native backend currently supports `asso_func=iou`. Use the Python backend if you need the other OCSORT association functions from `boxmot/configs/trackers/ocsort.yaml`.

::: boxmot.trackers.bbox.ocsort.OcSort
