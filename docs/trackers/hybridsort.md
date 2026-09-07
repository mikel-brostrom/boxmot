# HybridSort

[Paper: Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking](https://arxiv.org/abs/2308.00783)

Hybrid-SORT argues that MOT pipelines lean too heavily on strong cues such as appearance and overlap, even though those cues often fail together during heavy occlusion. The paper supplements them with weaker but cheap signals such as velocity direction, confidence, and height state, then combines those cues in a training-free online tracker. That makes the method attractive when association needs extra structure without turning into a much heavier offline system.

## What BoxMOT Needs For HybridSort

- A detector plus appearance embeddings when `use_embeddings=True`. The Python
  implementation can generate missing embeddings from a supplied `Frame` or
  consume embeddings already attached to `Detections`.
- Supports both AABB and OBB detections in BoxMOT.
- A good fit when you want richer association than OC-SORT or BoT-SORT-style matching, especially on crowded MOT benchmarks.

Direct construction accepts the shared `reid_model`, `reid_weights`, `device`,
`half`, and `reid_preprocess` options described in the
[Python API](../python/index.md#live-embeddings-in-reid-enabled-trackers).

::: boxmot.HybridSort
