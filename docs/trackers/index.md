---
description: Compare BoxMOT tracker backends and choose the right tracker for speed, ReID usage, and OBB support.
---

# Trackers

BoxMOT ships with multiple tracker backends so you can trade off speed, appearance matching, camera motion handling, and OBB support without changing the rest of your pipeline.

## Tracker Matrix

| Tracker | ReID / appearance | OBB | Typical fit |
| --- | --- | --- | --- |
| [ByteTrack](bytetrack.md) | No | Yes | Fast motion-only baseline |
| [BotSort](botsort.md) | Yes | Yes | Strong general-purpose tracker |
| [DeepOcSort](deepocsort.md) | Yes | No | Appearance-assisted OC-SORT variant |
| [StrongSort](strongsort.md) | Yes | No | Strong appearance matching with DeepSORT-style behavior |
| [OcSort](ocsort.md) | No | Yes | Motion-focused tracker with observation-centric updates |
| [HybridSort](hybridsort.md) | Yes | No | Hybrid association with long-term ReID options |
| [BoostTrack](boosttrack.md) | Optional / Yes | No | Recent appearance-assisted tracker with extra boosting terms |
| [SFSORT](sfsort.md) | No | Yes | Very fast tracker, including OBB workflows |

## How BoxMOT Selects a Tracker

Choose a tracker with `--tracker <name>`:

```bash
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker botsort --source video.mp4
boxmot eval --benchmark mot17-ablation --tracker boosttrack
```

When you select a tracker, BoxMOT loads `boxmot/configs/trackers/<name>.yaml`.

That YAML is used in two ways:

- `track` and `eval` read the `default` values
- `tune` reads the parameter ranges and options as the search space

## Practical Selection Guide

- Start with `bytetrack` when you want a fast baseline and do not need appearance embeddings.
- Use `botsort` when you want a balanced default with ReID and OBB support.
- Use `strongsort` or `deepocsort` when appearance matching quality matters more than raw throughput.
- Use `boosttrack` or `hybridsort` when you want newer appearance-assisted association variants.
- Use `sfsort` when you care about speed and want a motion-only tracker that also supports OBB.
- Use `ocsort` when you want a simple motion tracker with observation-centric updates and OBB support.

## OBB Support

The following trackers currently support OBB detections in BoxMOT:

- `bytetrack`
- `botsort`
- `ocsort`
- `sfsort`

OBB support requires both:

1. an OBB detector output with shape `(N, 7)`
2. a tracker from the list above

## Next Steps

Open an individual tracker page below to see CLI examples, BoxMOT-specific fit, and the generated API reference.
