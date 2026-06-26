# Tracker Overview

BoxMOT ships multiple tracker backends behind one interface.

## Current tracker set

| Tracker | Uses ReID | OBB support | Native C++ live | Native C++ replay |
| --- | --- | --- | --- | --- |
| ByteTrack | No | Yes | Yes | Yes |
| BotSort | Yes | Yes | Yes | Yes |
| StrongSort | Yes | No | No | No |
| OcSort | No | Yes | Yes | Yes |
| DeepOcSort | Yes | No | No | No |
| HybridSort | Yes | No | No | No |
| BoostTrack | Yes | No | No | No |
| OccluBoost | Yes | No | Yes | Yes |
| SFSORT | No | Yes | Yes | Yes |

## How to choose

- Start with `bytetrack` when you want a fast motion-only baseline.
- Use `botsort`, `strongsort`, `deepocsort`, `hybridsort`, or `boosttrack` when appearance cues matter.
- Use an OBB-capable tracker when your detector emits oriented boxes.
- Use `--tracker-backend cpp` for native C++ implementations when the selected tracker has a native backend.

## Config and factory

- Tracker runtime defaults come from `boxmot/configs/trackers`.
- The runtime factory lives in `boxmot/trackers/registry.py`.
- Native C++ tracker sources live under `boxmot/native/cpp/trackers/<name>/` and are registered from `boxmot/native/registry.py`.

Use [Native C++ Integration](../native/index.md) when you want to compile and embed a tracker directly in a C++ program.

Use the pages below for each tracker's API reference.
