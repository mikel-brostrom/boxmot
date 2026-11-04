# SFSORT

[Paper: SFSORT: Scene Features-based Simple Online Real-Time Tracker](https://arxiv.org/abs/2404.07553)

SFSORT associates boxes without a Kalman filter or an appearance encoder.
Its two matching stages use box similarity, while image regions distinguish
the retention of tracks lost near the boundary from those lost centrally.

## Tasks and variants

| Task | Default variant | Other variants |
| --- | --- | --- |
| [2D boxes (AABB)](../tasks/aabb.md) | [Motion-only](../tasks/aabb.md#motion-only) | None (ReID is unsupported) |
| [Oriented boxes (OBB)](../tasks/obb.md) | [Motion-only](../tasks/obb.md#motion-only) | None (ReID is unsupported) |

Python AABB mode also supports optional [EdgeTAM mask guidance](../tasks/masks.md#use-temporal-masks-in-association)
with `--tracker sfsort --tracker-backend python --asso-func iou --edgetam --mask-guidance-weights edgetam.pt`.
It requires IoU association and `per_class=False`, retains this tracker's
existing association rules, and adds temporal model inference. Accuracy
gains have not been established for this extension.

## What BoxMOT Needs For SFSORT

- Detector only. ReID is not required.
- Supports both AABB and OBB detections in BoxMOT.
- Best when throughput matters most and you want a very lightweight online tracker.

SFSORT also needs image dimensions for its scene regions. Supply a `Frame` or
NumPy image on every update, or set both positive `frame_width` and
`frame_height` in `SFSORTConfig`. For example,
`SFSORT(config=SFSORTConfig(frame_width=1920, frame_height=1080))` can process
detections without a frame when mask guidance is disabled. The ordinary
tracker uses dimensions, not image pixels. Both Python and native SFSORT use
fixed steps and reject variable timing and Kalman settings. Capture timestamps
remain metadata; AABB association uses observed boxes, OBB angle smoothing runs
per update, and region timeouts count updates.

`marginal_timeout` and `central_timeout` count updates and both default to zero.
Increase them explicitly if tracks lost near image edges or in the central
region should remain available for recovery. `dynamic_tuning` is off by default.

## Native C++ Backend

BoxMOT also ships a native C++17 SFSORT implementation under `boxmot/native/cpp/trackers/sfsort/`. It supports:

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
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker sfsort --tracker-backend cpp
boxmot track --tracker sfsort --tracker-backend cpp --source 0
```

## Python API

Pass algorithm settings through `SFSORTConfig`, for example
`SFSORT(config=SFSORTConfig(high_th=0.6))`. Direct construction and
`create_tracker("sfsort")` use the same algorithm defaults. See the
[tracker configuration guide](../config/trackers.md) for presets and component settings.

::: boxmot.SFSORTConfig

::: boxmot.SFSORT
