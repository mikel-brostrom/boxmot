# Component API Reference

## Canonical structures

Import canonical values from `boxmot.structures`. Their implementation modules
group 2D and 3D values by role:

| Module | Structures |
| --- | --- |
| [Geometry](../reference/boxmot/structures/geometry.md) | `Boxes`, `OrientedBoxes`, `Boxes3D` |
| [Detections](../reference/boxmot/structures/detections.md) | `Detections`, `Detections3D` |
| [Tracks](../reference/boxmot/structures/tracks.md) | `Tracks`, `Tracks3D`, `MultimodalTracks` |
| [Camera](../reference/boxmot/structures/camera.md) | `CameraModel` |

## Detection

::: boxmot.detectors.protocols.Detector

::: boxmot.detectors.protocols.DetectorCapabilities

::: boxmot.detectors.specs.DetectorSpec

::: boxmot.detectors.factory.create_detector

## Segmentation

::: boxmot.segmentors.protocols.Segmentor

::: boxmot.segmentors.specs.SegmentorSpec

::: boxmot.segmentors.factory.create_segmentor

## Appearance encoding

::: boxmot.reid.protocols.AppearanceEncoder
    options:
      skip_local_inventory: true

::: boxmot.reid.protocols.EncoderRequirements

::: boxmot.reid.specs.ReIDConfig
    options:
      skip_local_inventory: true

::: boxmot.reid.specs.ReIDEncoderSpec

::: boxmot.reid.factory.create_reid_encoder

## Tracking

::: boxmot.trackers.common.protocols.Tracker

::: boxmot.trackers.common.protocols.ReIDConfigurableTracker

::: boxmot.trackers.common.protocols.TrackerRequirements

::: boxmot.trackers.common.specs.TrackerSpec

::: boxmot.trackers.common.factory.create_tracker

Trackers accept canonical `Detections` or, for standalone box-only calls, exact
NumPy AABB `N x 6` / OBB `N x 7` rows. An optional canonical `Frame` may be
passed separately. NumPy input returns packed `float64` AABB `M x 8` / OBB
`M x 9` rows; canonical input returns `Tracks`. Appearance and mask inference
normally belong upstream in a `PerceptionPipeline`. Every high-level
ReID-enabled tracker adapter can also lazily invoke ReID when its input has no
embeddings and a `Frame` is supplied. Attached embeddings bypass that internal
backend. Native adapters then pass the resolved features to their model-free
C++ tracker libraries.

## Batched Kalman filters

Python trackers automatically batch compatible Kalman predictions and matched
corrections within each association stage. No CLI option is needed. Each track
retains its own covariance, calibrated noise, observation history, and adaptive
noise state.

For direct filter use, XYAH and XYWH provide `multi_predict`, `multi_project`,
and `multi_update`. Pass means shaped `(N, state_dimensions)` and covariances
shaped `(N, state_dimensions, state_dimensions)`; corrections also accept
measurements shaped `(N, measurement_dimensions)` and one confidence per row.
These methods return arrays without modifying the filter's stored state.

XYSR, XYSCR, and XYHR provide `predict_many(filters, ...)` and
`update_many(filters, measurements, ...)` for independent stateful filter
instances. These methods modify each instance while batching the matrix work.
Use these methods when a filter owns its state, noise, or observation history.
EagerMOT's `Kalman3D` similarly provides `multi_predict(filters)` and
`multi_update(filters, boxes)`.

Batching preserves elapsed-time and oriented-box handling. Missing observations
and observation-centric recovery retain each track's sequential history replay;
per-track bookkeeping and adaptive-noise updates still run independently.

## Batched camera-motion compensation

BoT-SORT, StrongSORT, BoostTrack, OccluBoost, DeepOCSORT, and HybridSORT batch
camera-motion updates automatically when CMC is enabled. No additional CLI
option is needed. The shared kernels transform box geometry, numerical
Jacobians, velocities, and covariances in arrays. Observation-centric trackers
also batch their retained observations and recovery states, preserving each
tracker's existing compensation policy.

Camera motion is estimated once per frame. Choose `sof` for translation,
rotation, and scale changes, or `ecc` for translation. These are the recommended
fast methods. ORB and SIFT remain available as slower opt-in methods in the Python
backend: select `cmc_method="orb"` or `cmc_method="sift"` in a Python tracker
configuration that supports CMC. Their useful settings can exceed the
one-millisecond latency target. Native backends support ECC and SOF only.

ECC and SOF resize before grayscale conversion and cap their default working
width regardless of source resolution. SOF uses a 192-pixel width, up to 48
background corners, and an 11-by-11 optical-flow window with at most 10 iterations.
It retains RANSAC inliers between feature refreshes (at most five frames), checking
the current detection mask on every frame. ECC uses a 96-pixel width and at most
10 iterations. Small inputs are not enlarged. Native BoT-SORT and OccluBoost
use the same settings. Explicit estimator scale/width overrides can increase cost.

The ECC and SOF defaults target one millisecond per frame; latency depends on
hardware, scene content, OpenCV threads, and detection count. Measure warmed calls including
preprocessing and state updates, alongside known-motion accuracy:

```bash
uv run --no-sync python -m tests.performance.trackers.motion.benchmark_cmc_latency \
  --widths 1280 1920 3840 --iterations 100 --json /tmp/cmc-latency.json
```

The benchmark defaults to ECC and SOF. Compare the slower opt-in methods with:

```bash
uv run --no-sync python -m tests.performance.trackers.motion.benchmark_cmc_latency \
  --methods orb sift --widths 1920 --iterations 100
```

This benchmark uses one OpenCV thread and preloaded BGR frames without detection
masks. It reports mean, median, p95 and maximum latency plus original-image pixel
error. Add `--check-budget` to fail when any measured p95 exceeds one millisecond.
ECC's translation model cannot recover the benchmark's rotation/scale component.
Re-run joint camera/Kalman selection for older fitted profiles: the estimator
change advances the analysis rule to `sof-motion-v4`.

Frame pairs remain sequential. General affine and projective OBB fitting still
uses one OpenCV rectangle fit per box, and mask rasterization retains individual
polygon calls to preserve overlaps.

Compare cached replay against a saved package snapshot without rerunning
detection or ReID:

```bash
uv run --no-sync python -m tests.performance.trackers.benchmark_cached_cmc \
  --baseline /path/to/snapshot-containing-boxmot \
  --build /path/to/existing/materialization \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --tracker botsort --workers 7 --repeat 3 \
  --output /tmp/cmc-comparison
```

Use `--mode preloaded --sequence MOT17-04-FRCNN --frames 100` to exclude image
decoding from the measured tracking loop. Both modes check track IDs and
geometry; complete replay also checks the evaluation metrics.
