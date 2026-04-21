# Tracker Overview

BoxMOT ships multiple tracker backends behind one interface.

## Current tracker set

| Tracker | Uses ReID | OBB support |
| --- | --- | --- |
| ByteTrack | No | Yes |
| BotSort | Yes | Yes |
| StrongSort | Yes | No |
| OcSort | No | Yes |
| DeepOcSort | Yes | No |
| HybridSort | Yes | No |
| BoostTrack | Yes | No |
| SFSORT | No | Yes |

## How to choose

- Start with `bytetrack` when you want a fast motion-only baseline.
- Use `botsort`, `strongsort`, `deepocsort`, `hybridsort`, or `boosttrack` when appearance cues matter.
- Use an OBB-capable tracker when your detector emits oriented boxes.

## Config and factory

- Tracker runtime defaults come from `boxmot/configs/trackers`.
- The runtime factory lives in `boxmot/trackers/tracker_zoo.py`.

Use the pages below for each tracker's API reference.
