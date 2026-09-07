# Track

`track` runs a source through engine-owned perception, tracking, and output
sinks.

```bash
boxmot track \
  --source video.mp4 \
  --detector yolov8n \
  --reid osnet_x0_25_msmt17 \
  --tracker botsort \
  --save
```

## Inference sources

Sources may be images, directories, finite videos, webcams, or URLs. The source
owns OpenCV capture, BGR-to-RGB conversion, stride, timestamps, reconnect
policy, and stable sequence/frame identity. A runner resets state between
sequences and forwards each `PipelineResult` to configured rendering, video,
MOT text, JSON, composite, or null sinks.

## Geometry

Select a fixed tracker geometry with `--geometry aabb|obb`. Detector output and
tracker mode must agree:

```bash
boxmot track \
  --source aerial.mp4 \
  --detector yolo11n-obb.pt \
  --geometry obb \
  --tracker ocsort
```

Canonical OBB angles are radians and remain unwrapped for temporal continuity.
Pipeline values remain typed; `to_obb_rows()` produces the 9-column output
layout when an external boundary needs it.

## Masks and appearance

The resolved tracker declares whether it needs embeddings, masks, or frame
pixels. The pipeline adds only missing requirements:

- A detector-native mask or embedding payload is preserved.
- A configured segmentor runs only when masks are needed and absent.
- A configured appearance encoder runs only when embeddings are needed and
  absent.
- Without an external encoder, a ReID-enabled tracker adapter receives the
  `Frame` and extracts its private embeddings internally.
- Transitive requirements are honored: a mask-aware encoder triggers
  segmentation first.

Sam2Mot requires full-frame detection-aligned foreground masks. Trackers with
`use_embeddings` consume an upstream payload when available. For a non-empty
batch without embeddings, every ReID-enabled tracker adapter can instead lazily
generate them from the supplied `Frame`; attached embeddings bypass internal
inference, and empty batches do not initialize the model. Native adapters pass
the resulting features to the model-free C++ tracker library through its typed
ABI.

Tracker-private embeddings are used only for association. Requesting
`PipelineOutputs(embeddings=True)` still requires a detector-provided payload
or an external appearance encoder because the private matrix is not added to
the returned `Detections`.

## Sequence state

A `TrackingPipeline` processes exactly one sequence. When frame indices are
present, they must increase. Reset the pipeline before starting another
sequence. The runner handles this for engine-owned sources; embedded Python
callers invoke `pipeline.reset()` themselves.

## Python

Use factories and structures when composing a pipeline:

```python
from boxmot import create_tracker
from boxmot.pipelines import TrackingPipeline
from boxmot.trackers import TrackerSpec

tracker = create_tracker(TrackerSpec(name="bytetrack", geometry="aabb"))
pipeline = TrackingPipeline(detector=detector, tracker=tracker)

for frame in frames:  # Sequence[boxmot.structures.Frame]
    result = pipeline.step(frame)
    consume(frame, result.tracks)
```

For externally supplied detections, construct the pipeline with
`detector=None` and call `step_detections(frame, detections)`.

For an appearance-enabled tracker adapter, `TrackingPipeline` may omit `reid`;
the tracker then extracts missing embeddings from each live frame. Pass a
shared encoder as `reid` when the detections returned in `PipelineResult` must
include embeddings or when several consumers reuse the same encoder.

A standalone box-only tracker may instead receive an exact NumPy AABB6 or OBB7
matrix directly and returns packed `float64` AABB8 or OBB9 rows. Use
`Detections` for `Tracks`, embeddings, masks, sample metadata, and all pipeline
calls.

## Working with results

`PipelineResult` contains exactly `detections` and `tracks`. Geometry lives in
`result.tracks.geometry.values`; IDs, scores, classes, and detection indices are
separate aligned tensors. Timing is runner metadata keyed by sample ID, not a
result field.

## Shutdown report

Normal completion, `q`/Escape in the display window, and Ctrl-C all close the
source and output sinks before showing the final Rich report. The report keeps
partial measurements from an interrupted frame and separates detector and
ReID preprocessing, inference, and postprocessing. It also reports component
totals, source acquisition, pipeline enrichment and validation, tracker
association/update, rendering, sink I/O, residual overhead, and overall
latency with totals, per-frame averages, and FPS.

When `--save` and `--show` are used together, the video writer and display
share one rendered frame rather than rendering the same result twice.

## Native trackers

Select a registered native implementation with `--tracker-backend cpp`.
Factory validation rejects unsupported masks, per-class mode, or geometry
before tracking starts. Native and Python implementations share the structured
public contract. ReID-enabled native adapters generate missing embeddings from
the frame just like their Python counterparts; the underlying C++ tracker
libraries consume the resulting typed buffers and do not load models.

## Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.track
    :command: track
    :prog_name: boxmot track
    :depth: 0
