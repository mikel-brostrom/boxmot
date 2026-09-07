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
authored experiment resolution lives in `boxmot.engine.experiment_config` and
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
  -> PipelineResult(detections, tracks)
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
  -> PipelineResult(detections, tracks)
```

`TrackingPipeline.step_detections(frame, detections)` serves HTTP requests and
cached replay. It does not bypass validation or invoke a detector.

## State

A pipeline instance tracks exactly one sequence. If frame indices are supplied,
they must increase. Interleaving another sequence is rejected until `reset()`;
reset also clears tracker state.

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
