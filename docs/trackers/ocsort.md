# OcSort

[Paper: Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking](https://arxiv.org/abs/2203.14360)

OC-SORT focuses on a specific failure mode in Kalman-filter trackers: error accumulation during occlusion and non-linear motion. The paper replaces a purely prediction-centric view with an observation-centric one, using detector observations to reconstruct a more reliable virtual trajectory across missed frames. That makes the tracker much more robust than vanilla SORT in crowded scenes while keeping the same simple online structure.

Python AABB mode also supports optional [EdgeTAM mask guidance](../tasks/masks.md#use-temporal-masks-in-association)
with `--tracker ocsort --tracker-backend python --asso-func iou --edgetam --mask-guidance-weights edgetam.pt`.
It requires IoU association and `per_class=False`, retains this tracker's
existing association rules, and adds temporal model inference. Accuracy
gains have not been established for this extension.

## What BoxMOT Needs For OcSort

- Detector only. ReID is not required.
- Supports both AABB and OBB detections in BoxMOT.
- A strong choice when you want a fast motion-only tracker but expect more non-linear motion or occlusion than ByteTrack handles comfortably.

## Native C++ Backend

BoxMOT ships a native C++17 OcSort implementation under `boxmot/native/cpp/trackers/ocsort/`. It supports:

- cached `eval` and `tune` streamed through the live typed API
- live `track` through `--tracker-backend cpp`
- both AABB and OBB detection layouts in the native tracker path

Requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker ocsort --tracker-backend cpp
boxmot track --tracker ocsort --tracker-backend cpp --source 0
```

The native backend honors `asso_func` from
`boxmot/configs/trackers/ocsort.yaml`. AABB and OBB tracking support `iou`,
`giou`, `diou`, `ciou`, `hmiou`, and `centroid`. Centroid association uses the
first live image to initialize and cache the frame dimensions. OBB `ciou` is a
custom experimental aspect-ratio adaptation, while OBB `hmiou` is an
experimental global-y height cue intended only for scenes where image vertical
is meaningful. See the [association function guide](../config/trackers.md#association-function)
for the exact OBB semantics.

## Python API

Pass algorithm settings through `OcSortConfig`, for example
`OcSort(config=OcSortConfig(det_thresh=0.5))`. Direct construction and
`create_tracker("ocsort")` use the same algorithm defaults. See the
[tracker configuration guide](../config/trackers.md) for presets and component settings.

::: boxmot.OcSortConfig

::: boxmot.OcSort
