# StrongSort

[Paper: StrongSORT: Make DeepSORT Great Again](https://arxiv.org/abs/2202.13514)

StrongSort uses an appearance gallery, Kalman motion, camera-motion
compensation, and geometric fallback association. BoxMOT runs its online
tracker; detector and appearance-model selection are configured separately.

## Tasks and variants

| Task | Default variant | Other variants |
| --- | --- | --- |
| [2D boxes (AABB)](../tasks/aabb.md) | [AABB + ReID](../tasks/aabb.md#appearance-assisted-reid) | None (ReID is required) |
| [Oriented boxes (OBB)](../tasks/obb.md) | [OBB + ReID](../tasks/obb.md#appearance-assisted-reid) | None (ReID is required) |

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

ECC camera-motion compensation is enabled by default. Supply the current frame
even when detections already contain embeddings. Set
`StrongSortConfig(use_cmc=False)` to disable CMC, or select another supported
`cmc_method`. Appearance remains required; missing embeddings still need image
pixels for ReID. Centroid association and optional mask guidance have their own
frame requirements.

Tracks normally need `n_init=3` consecutive observations for confirmation.
`max_age` retains unmatched tracks internally; it does not guarantee a visible
output row on every missing observation. This page describes the online tracker.
The paper's offline linking and interpolation stages are separate workflows;
see [evaluation postprocessing](../guides/evaluation.md).

## Python API

Pass algorithm settings through `StrongSortConfig`, for example
`StrongSort(config=StrongSortConfig(min_conf=0.6))`. Direct construction and
`create_tracker("strongsort")` use the same algorithm defaults. See the
[tracker configuration guide](../config/trackers.md) for presets and component settings.

::: boxmot.StrongSortConfig

::: boxmot.StrongSort
