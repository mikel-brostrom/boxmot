# StrongSort

[Paper: StrongSORT: Make DeepSORT Great Again](https://arxiv.org/abs/2202.13514)

StrongSORT revisits DeepSORT and shows that a stronger baseline matters. The paper improves the detector and appearance encoder, adds better motion handling and camera compensation, and then layers on lightweight postprocessing ideas to recover missed links and detections. The core message is that a carefully engineered DeepSORT-style tracker can remain competitive without changing the online MOT formulation.

Python AABB mode also supports optional [EdgeTAM mask guidance](../tasks/masks.md#use-temporal-masks-in-association)
with `--tracker strongsort --tracker-backend python --asso-func iou --edgetam --mask-guidance-weights edgetam.pt`.
It requires IoU association and `per_class=False`, retains this tracker's
existing association rules, and adds temporal model inference. Accuracy
gains have not been established for this extension.

## What BoxMOT Needs For StrongSort

- A detector plus appearance embeddings. Appearance cues are central to this
  tracker. The Python implementation can generate missing embeddings from a
  supplied `Frame` or consume embeddings already attached to `Detections`.
- Supports both AABB and OBB detections in BoxMOT.
- Good when appearance matching matters more than raw speed, especially for pedestrian-style MOT benchmarks.

Direct construction accepts `reid=ReIDConfig(...)` or a prebuilt
`AppearanceEncoder`, as described in the
[Python API](../python/index.md#live-embeddings-in-reid-enabled-trackers).

## Python API

Pass algorithm settings through `StrongSortConfig`, for example
`StrongSort(config=StrongSortConfig(min_conf=0.6))`. Direct construction and
`create_tracker("strongsort")` use the same algorithm defaults. See the
[tracker configuration guide](../config/trackers.md) for presets and component settings.

::: boxmot.StrongSortConfig

::: boxmot.StrongSort
