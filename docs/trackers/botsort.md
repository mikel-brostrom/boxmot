# BotSort

[Paper: BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651)

BoT-SORT extends the ByteTrack family by combining motion, appearance, and camera-motion compensation more explicitly. The paper improves the Kalman state, uses global motion compensation, and fuses ReID cues with IoU-based association to make identity assignment more stable in crowded scenes and moving-camera footage. The result is a tracker that is still online and practical, but more robust than motion-only alternatives when identities are ambiguous.

## What BoxMOT Needs For BotSort

- A detector plus appearance embeddings when `use_embeddings=True`. Both the
  Python implementation and native adapter can generate missing embeddings
  from a supplied `Frame` or consume embeddings already attached to
  `Detections`.
- Supports both AABB and OBB detections in BoxMOT.
- Best when you need stronger identity preservation than ByteTrack, especially with camera motion or repeated occlusions.

Direct construction accepts the shared `reid_model`, `reid_weights`, `device`,
`half`, and `reid_preprocess` options described in the
[Python API](../python/index.md#live-embeddings-in-reid-enabled-trackers).

## Native C++ Backend

BoxMOT also ships a native C++17 BotSort implementation under `boxmot/native/cpp/trackers/botsort/`. It supports:

- cached `eval` and `tune` streamed through the live typed API
- live `track` through `--tracker-backend cpp`
- both AABB and OBB detections for live tracking and cached evaluation
- typed generated or precomputed embeddings supplied through the same v2 update ABI
- model-free C++ tracker code; optional ReID inference is owned by its Python adapter

Requirements:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker botsort --tracker-backend cpp
boxmot track --tracker botsort --tracker-backend cpp --reid models/lmbn_n_duke.pt --source 0
```

The native BoT-SORT adapter bypasses its encoder when canonical `Detections`
already carry embeddings. Otherwise, a non-empty batch can generate them
lazily from its `Frame`, then pass the typed feature buffer to the C++ tracker.
The C++ library itself never loads or runs a ReID model. Python BoT-SORT follows
the same generated-or-attached contract. Materialized builds record the encoder
fingerprint and publish embeddings as keyed Parquet rows. Evaluation streams
those rows through the same live typed API used by track mode; there is no
positional replay executable. See
[Native C++ Integration](../native/index.md#capabilities-and-requirements).

::: boxmot.BotSort
