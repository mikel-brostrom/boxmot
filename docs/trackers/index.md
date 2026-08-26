# Tracker Overview

BoxMOT ships multiple tracker backends behind one interface.

## Current tracker set

| Tracker | Uses ReID | Uses masks | OBB support | Native C++ live | Native C++ replay |
| --- | --- | --- | --- | --- | --- |
| ByteTrack | No | No | Yes | Yes | Yes |
| BotSort | Yes | No | Yes | Yes | Yes |
| StrongSort | Yes | No | Yes | No | No |
| OcSort | No | No | Yes | Yes | Yes |
| DeepOcSort | Yes | No | Yes | No | No |
| HybridSort | Yes | No | Yes | No | No |
| BoostTrack | Yes | No | Yes | No | No |
| OccluBoost | Yes | No | Yes | Yes | Yes |
| SFSORT | No | No | Yes | Yes | Yes |
| [SAM2MOT](sam2mot.md) | No | Yes | Yes | No | No |

## How to choose

- Start with `bytetrack` when you want a fast motion-only baseline.
- Use `botsort`, `strongsort`, `deepocsort`, `hybridsort`, `boosttrack`, or `occluboost` when appearance cues matter.
- Use `sam2mot` when each detection has a row-aligned segmentation mask and you want mask-aware association without ReID.
- All registered Python trackers accept both AABB and OBB detections.
- Use `--tracker-backend cpp` for native C++ implementations when the selected tracker has a native backend.

## Config and factory

- Tracker runtime defaults and tuning search spaces share `boxmot/configs/trackers/<tracker>.yaml`; reusable scalar presets remain under `boxmot/configs/trackers/presets`.
- The runtime factory lives in `boxmot/trackers/registry.py`.
- Native C++ tracker sources live under `boxmot/native/cpp/trackers/<name>/` and are registered from `boxmot/native/registry.py`.

Use [Native C++ Integration](../native/index.md) when you want to compile and embed a tracker directly in a C++ program.

Use the pages below for each tracker's API reference.
