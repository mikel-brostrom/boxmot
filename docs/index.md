# BoxMOT

BoxMOT is a structured tracking-by-detection toolkit for axis-aligned boxes,
oriented boxes, and instance masks. Its dependency direction is explicit:

```text
structures -> domain components -> pipelines -> engine
```

Detectors, segmentors, appearance encoders, and trackers are independently
usable. Pipelines normally provide reusable masks and appearance embeddings.
For live use, every high-level ReID-enabled tracker adapter can also extract
missing embeddings lazily from a supplied frame. Native adapters pass those
features to their model-free C++ tracker libraries. The engine owns media
sources, sinks, services, CLI workflows, evaluation, tuning, and resumable
materialization.

[MafHda](trackers/maf_hda.md) combines motion and masked correlation-filter
appearance for AABB detections with nonempty full-frame instance masks and
current image frames.
`boxmot eval-trackrcnn --tracker maf_hda` evaluates saved TrackR-CNN predictions
on KITTI MOTS; see the MAF-HDA tracker page for the required paths.
[EagerMot](trackers/eagermot.md) adds 2D/3D sensor fusion through the Python
tracker API with independent detection batches and camera calibration.
Use `boxmot eval-eagermot` to evaluate saved KITTI PointGNN and TrackR-CNN
predictions against KITTI MOTS masks; the tracker page provides the command.
[`boxmot tune-eagermot`](trackers/eagermot.md#tune-separate-class-profiles)
optimizes separate car and pedestrian profiles together for class-average mask
HOTA and saves profiles for `eval-eagermot --class-config`.

## Get started

```bash
pip install boxmot
boxmot --help

boxmot track \
  --source video.mp4 \
  --detector yolov8n \
  --tracker bytetrack \
  --save
```

For repeatable evaluation, publish perception once and name the build on every
downstream command:

```bash
boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml
boxmot eval --experiment mot17/ablation-yolox-lmbn.yaml --build BUILD_ID
```

## Where to go next

| Goal | Guide |
| --- | --- |
| Install optional detector, service, or export runtimes | [Installation](getting-started/installation.md) |
| Run the CLI | [CLI](usage/index.md) |
| Track live or finite media | [Track](modes/track.md) |
| Build reusable keyed perception data | [Materialize](modes/materialize.md) |
| Evaluate, tune, or research against a fixed build | [Modes](modes/index.md) |
| Compose Torch-native components in Python | [Python API](python/index.md) |
| Select an association algorithm | [Trackers](trackers/index.md) |
| Embed a service or native tracker | [Integrations](integrations/index.md) |

Canonical structures are CPU-contiguous and validated without implicit
conversion. Standalone box-only tracker calls may use exact NumPy AABB6 or OBB7
rows and receive packed NumPy AABB8 or OBB9 rows; pipelines and enriched
detections use canonical structures.
