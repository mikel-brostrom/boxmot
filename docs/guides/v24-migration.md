# Migrating to v24

Version 24.0.0 is one coordinated public cutover. It intentionally does not
provide compatibility wrappers for the old Python, CLI, or cache interfaces.

## Breaking-change map

| Before v24 | v24 replacement |
| --- | --- |
| `BoxMOT(...)` workflow facade | Explicit component factories plus `PerceptionPipeline` or `TrackingPipeline`; use the engine CLI for source-to-sink workflows |
| Root `Detector` | `boxmot.detectors.DetectorSpec` and `create_detector` |
| Root `ReIDModel` | `boxmot.reid.ReIDEncoderSpec` and `create_reid_encoder` |
| `create_tracker("name", reid_weights=..., device=...)` | `create_tracker(TrackerSpec(...))`; trackers contain no model or device configuration |
| `tracker.update(numpy_rows, img=..., embs=..., masks=...)` | Attach enrichments to `Detections`, then call `tracker.update(detections, frame)` |
| `DetectionBatch`, `TrackResults`, `FrameData`, `FramePayload`, engine result records | `Frame`, `Boxes`/`OrientedBoxes`, `MaskBatch`, `Detections`, `Tracks`, and `PipelineResult` |
| `boxmot.api` | Removed; compose public domain packages directly |
| `boxmot.data` | `boxmot.datasets` for immutable build loading and dataset configuration; engine materialization owns finite sources |
| `boxmot generate` | `boxmot materialize` |
| Eval/tune/research selecting or creating caches | Eval can materialize an experiment when `--build` is omitted; dataset-only eval, tune, and research require `--build PATH_OR_ID` |
| Positional `.npy`/`.npz` caches | Keyed `boxmot.dataset/v1` Parquet builds joined by `sample_id` and `instance_id` |
| Native replay executable and caller-sized float row buffers | Live typed v2 C ABI with `int64` identifiers and library-owned output/free |

## Root imports

The package root exports exactly `__version__`, `create_tracker`, and the ten
lazy tracker classes: `BoostTrack`, `BotSort`, `ByteTrack`, `DeepOcSort`,
`HybridSort`, `OccluBoost`, `OcSort`, `Sam2Mot`, `SFSORT`, and `StrongSort`.
Import structures, specs, protocols, factories, pipelines, and datasets from
their named subpackages.

## Boundary conversion

Canonical tensors are CPU-contiguous. Constructors validate without silently
converting, moving, clipping, or filtering. Convert external framework values
once before construction:

```python
import torch

from boxmot.structures import Boxes, Detections

detections = Detections(
    geometry=Boxes(torch.as_tensor(xyxy, dtype=torch.float32).cpu().contiguous()),
    scores=torch.as_tensor(scores, dtype=torch.float32).cpu().contiguous(),
    class_ids=torch.as_tensor(class_ids, dtype=torch.int64).cpu().contiguous(),
    sample_id=sample_id,
)
tracks = tracker.update(detections)
```

Use row serializers only where an external file or wire format requires them.

## Cache cutover

Materialize again under the v1 schema. Existing pre-v1 positional `.npy` and
`.npz` cache directories remain physically untouched and unsupported. A
compatible v1 build in the former platform-cache build root may seed the shared
detection cache—or be imported directly when its full build identity matches—
when using the repository-local default.

```bash
boxmot eval --experiment EXPERIMENT
```

## HTTP compatibility

The `/v1` request and response wire format is preserved. The service now
decodes those rows into canonical structures internally and serializes tracks
only at the response boundary. Existing clients still need ordered frame IDs
and the same retry/session rules.
