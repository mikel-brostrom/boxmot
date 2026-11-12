# Tracking Pipeline

The v24 boundary points in one direction:

```text
Frame / Detections / Tracks
          |
          v
detector, segmentor, appearance encoder, tracker
          |
          v
PerceptionPipeline / TrackingPipeline
          |
          v
engine sources, sinks, runners, service, materialization, eval
```

Structures import no domain or engine packages. Domain components import
structures but never pipelines or engine code. Pipelines compose components
without owning I/O. The engine owns process lifetime and external boundaries.
Dataset configuration and persisted-format concerns live under
`boxmot.datasets`; finite source catalogs live with materialization, while
authored experiment resolution lives in `boxmot.engine.config.experiments` and
build compatibility lives with materialization. Artifact identity is shared
component infrastructure, while detector, segmentor, and ReID specification
resolution belongs to each corresponding domain package.

## Live detection path

```text
Frame
  -> Detector.predict([frame])
  -> Detections
  -> optional Segmentor.segment(...)
  -> optional AppearanceEncoder.encode(...)
  -> Tracker.update(detections, frame?)
       -> optional private ReID extraction in a ReID-enabled tracker adapter
  -> optional track-attribute sampling and completed-result snapshot
  -> PipelineResult(detections, tracks, attributes)
```

`TrackingPipeline.step(frame)` uses this path. Enrichment is driven by the
resolved tracker requirements, requested outputs, and transitive encoder
requirements. Detector-native enrichments are preserved. When appearance is a
tracker requirement and no external encoder supplies it, a ReID-enabled
tracker adapter can derive its private embeddings from the frame. This private
matrix does not become a `PipelineResult.detections` output; explicitly
requesting embeddings still requires upstream enrichment. A native adapter
computes the same fallback before passing a typed embedding buffer to its
model-free C++ tracker library.

## Supplied-detection path

```text
Frame + Detections
  -> same enrichment and validation
  -> Tracker.update(detections, frame?)
  -> optional track-attribute sampling and completed-result snapshot
  -> PipelineResult(detections, tracks, attributes)
```

`TrackingPipeline.step_detections(frame, detections)` serves HTTP requests and
cached replay. It does not bypass validation or invoke a detector.

Output enrichments and tracker inputs are separate. A pipeline can preserve
masks or embeddings in `PipelineResult.detections` for rendering or export
while stripping channels the configured tracker does not consume. A direct
`tracker.update()` call instead rejects unsupported enrichments.

Optional [track attributes](../config/attributes.md) use the original frame and
current tracks after tracking. `boxmot.attributes` owns crop sampling, Ollama/vLLM
requests, and its disk-backed queue; the pipeline exposes detached snapshots
without changing `Tracks`. Engine workflows own completion, output journaling,
and shutdown. Embedded callers manage the enricher's context and event API.

## Sensor fusion path

`TrackingPipeline` handles image tracking only. It rejects trackers requiring
3D detections or calibration at construction. Call
[EagerMot.update](../trackers/eagermot.md#sensor-fusion-example) directly with
independent 2D/3D batches and a camera model. With canonical image detections,
this call returns `MultimodalTracks`; NumPy input returns a pair of packed
arrays. The saved-sensor `eval` and `tune` workflows orchestrate whole sequences
and return `ValidationResult` and `TuneResult`, respectively.

## State

A pipeline instance tracks exactly one sequence. If frame indices are supplied,
they must increase. Interleaving another sequence is rejected until `reset()`;
reset also clears tracker state.

With attributes enabled, reset starts a new attribute generation. Late results
from an earlier generation remain completion events and cannot populate the new
sequence's snapshot, even when the tracker reuses the same numeric IDs.

Timing, progress, retries, error handling, cleanup, persistence, and display
are engine metadata and never fields of `PipelineResult`.

## Native path

Native trackers implement the same structured Python protocol through a typed
v2 C ABI. One update mutates state exactly once, returns a library-allocated
output object, and is followed by an explicit free. The engine does not retry a
state mutation after guessing an undersized output allocation.

Materialized replay reads stable keyed Parquet rows and calls this live API.
Positional NumPy replay files and dedicated cache executables are not part of
v24.
