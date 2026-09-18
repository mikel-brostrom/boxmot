# HybridSort

[Paper: Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking](https://arxiv.org/abs/2308.00783)

Hybrid-SORT argues that MOT pipelines lean too heavily on strong cues such as appearance and overlap, even though those cues often fail together during heavy occlusion. The paper supplements them with weaker but cheap signals such as velocity direction, confidence, and height state, then combines those cues in a training-free online tracker. That makes the method attractive when association needs extra structure without turning into a much heavier offline system.

Python AABB mode also supports optional [EdgeTAM mask guidance](../tasks/masks.md#use-temporal-masks-in-association)
with `--tracker hybridsort --tracker-backend python --asso-func iou --edgetam --mask-guidance-weights edgetam.pt`.
It requires IoU association and `per_class=False`, retains this tracker's
existing association rules, and adds temporal model inference. Accuracy
gains have not been established for this extension.

## What BoxMOT Needs For HybridSort

- A detector plus appearance embeddings when `use_embeddings=True`. The Python
  implementation can generate missing embeddings from a supplied `Frame` or
  consume embeddings already attached to `Detections`.
- Supports both AABB and OBB detections in BoxMOT.
- A good fit when you want richer association than OC-SORT or BoT-SORT-style matching, especially on crowded MOT benchmarks.

Direct construction accepts `reid=ReIDConfig(...)` or a prebuilt
`AppearanceEncoder`, as described in the
[Python API](../python/index.md#live-embeddings-in-reid-enabled-trackers).

Tracker YAML profiles and Python construction use lowercase option names:
`eg_weight_high_score`, `eg_weight_low_score`, `tcm_first_step`,
`tcm_byte_step`, and `tcm_byte_step_weight`. Tuned profiles use these same names.

In AABB mode, `tcm_first_step=False` disables the first-stage motion-direction
and confidence cues while retaining geometry and appearance matching.

## MOT17 ablation

HybridSORT's defaults reproduce the historical MOT17 baseline settings, so
the standard evaluation command uses them directly:

```bash
boxmot eval \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --tracker hybridsort --tracker-backend python
```

The `hybridsort-mot17-ablation` preset explicitly selects the same settings.
The previous global defaults were SportsMOT validation tuning; select
`--tracker-config hybridsort-sportsmot-val` to retain those settings, including
for the `sportsmot/val-yolox-lmbn.yaml` experiment. This restoration also changes
the defaults for direct Python construction, live tracking, and OBB tracking.

## Python API

Pass algorithm settings through `HybridSortConfig`, for example
`HybridSort(config=HybridSortConfig(det_thresh=0.5))`. Direct construction and
`create_tracker("hybridsort")` use the same algorithm defaults. See the
[tracker configuration guide](../config/trackers.md) for presets and component settings.

::: boxmot.HybridSortConfig

::: boxmot.HybridSort
