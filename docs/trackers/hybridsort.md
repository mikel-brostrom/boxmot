# HybridSort

HybridSort combines motion cues, camera motion compensation, and appearance features, including long-term ReID options. In BoxMOT it is useful when you want a more configurable hybrid association strategy than the simpler motion-only trackers.

## BoxMOT Fit

- ReID: yes
- OBB: not supported
- Best for: experiments that need richer association logic and long-term appearance options

## CLI

```bash
boxmot track --detector yolov8n --reid osnet_x0_25_msmt17 --tracker hybridsort --source video.mp4 --save
boxmot eval --benchmark mot17-ablation --tracker hybridsort
```

## Config

HybridSort is available as a tracker backend in BoxMOT, and its runtime defaults are loaded from `boxmot/configs/trackers/hybridsort.yaml`.

## API Reference

::: boxmot.trackers.hybridsort.hybridsort.HybridSort
