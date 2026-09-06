# Compare BoxMOT Components

Choose components against the constraints of your footage and deployment, then
compare them on the same cached detections. Tracker and ReID rankings are not
portable across detectors, datasets, hardware, or association settings.

## Choose a tracker

| Requirement | Good starting point | Why |
| --- | --- | --- |
| Small, fast motion-only baseline | ByteTrack or SFSORT | No appearance model is required. |
| Abrupt or non-linear motion | OcSort | Observation-centric updates reduce motion-model drift. |
| Moving camera | BotSort or DeepOcSort | Camera-motion compensation is available. |
| Crowds and longer occlusions | StrongSort, DeepOcSort, HybridSort, BoostTrack, or OccluBoost | Appearance features can reconnect identities. |
| Instance masks should affect association | Sam2Mot | Mask overlap participates in matching. |
| In-process native C++ | BotSort, ByteTrack, OcSort, OccluBoost, or SFSORT | These trackers have registered native live backends; cached workflows stream keyed rows through that same API. |

This is a starting-point guide, not a universal ranking. The
[tracker overview](../trackers/index.md) contains the support matrix, and each
tracker page explains its own requirements and tradeoffs.

## Compare ReID models

Appearance-assisted trackers trade more compute for another identity cue. Use
the [ReID model table](../config/reid-models.md) to compare parameters, compute,
embedding size, and available retrieval results. Compare runtime formats on
your target device as well: PyTorch, ONNX, TensorRT, OpenVINO, Core ML, and
TFLite can have different latency and deployment constraints.

## Run a fair tracker comparison

Materialize detections and embeddings once, then reuse the same immutable build
for each tracker:

```bash
boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml

boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker bytetrack
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker botsort
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID --tracker boosttrack
```

Keep the experiment, detector profile, split, postprocessing, and metric
configuration fixed. Record the tracker YAML and runtime overrides with every
result.

## Read the metrics together

- **HOTA** balances detection, association, and localization quality.
- **IDF1** emphasizes identity consistency.
- **MOTA** combines misses, false positives, and identity switches.
- **FPS and latency** must be measured on the deployment hardware and include
  any ReID or camera-motion stages you enable.

See [Evaluation and Postprocessing](../guides/evaluation.md) for metric details
and [Experiment Workflows](../guides/experiments.md) for cache identity and
reproducibility.
