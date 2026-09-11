# Evaluation and Postprocessing

Evaluation has two independent inputs:

1. A dataset adapter owns raw ground truth and its source-catalog digest.
2. A complete `boxmot.dataset/v1` build owns keyed perception artifacts.

```bash
boxmot eval \
  --dataset mot17 \
  --split ablation \
  --build runs/materializations/BUILD_ID \
  --data-root datasets/mot \
  --tracker boosttrack
```

The evaluator validates that those inputs describe the same source, split,
taxonomy, and geometry before replay. Experiment mode also validates semantic
component fingerprints. Omit `--build` to materialize or reuse a compatible
canonical build before replay:

```bash
boxmot eval \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --data-root datasets/mot \
  --tracker boosttrack
```

Supplying `--build` skips automatic preparation and evaluates exactly that
build.

Automatic preparation stores detector results independently from ReID output
under `<selected-build-root>/.cache/detect` (by default,
`runs/materializations/.cache/detect`). Experiments that share the source
dataset, detector, geometry, and class mapping therefore skip repeated detector
inference and run only their remaining derived stages. Detector-native masks or
embeddings bypass this geometry-only cache.

Tracker requirements are checked against published artifacts. For example, a
configuration with `use_embeddings: true` requires embeddings in the build;
MafHda requires AABB detections, nonempty full-frame masks, and current frames.
Missing inputs produce an actionable materialization error.

## KITTI MOTS evaluation

KITTI supports both images plus detections and images plus detections plus
predicted masks. Choose the metric geometry explicitly; a build containing
masks can still be evaluated using boxes.

| Evaluation | Option | Required predictions | Overlap measure |
| --- | --- | --- | --- |
| Boxes (default) | No extra flag | Detections | Box IoU |
| Segmentation | `--eval-masks` | Detections and masks | Mask IoU |

Follow the [KITTI MOTS download and setup instructions](../config/datasets.md#download-kitti-mots-data)
to obtain the color images and PNG annotations, then prepare a build for its
`val` split. This command evaluates boxes without requiring predicted
segmentations:

```bash
uv run --no-sync python -m boxmot.engine.cli eval \
  --dataset kitti-mots \
  --split val \
  --data-root datasets \
  --build runs/materializations/BUILD_ID \
  --tracker bytetrack
```

Here `datasets/KITTI-MOTS` contains the extracted images and instance PNGs,
and `BUILD_ID` is the ID returned by materializing your KITTI MOTS experiment.
For automatic preparation, use `--experiment /path/to/kitti-mots-val.yaml`
in place of `--dataset`, `--split`, and `--build`. The experiment must supply
the detections needed by the tracker.

Box evaluation derives tight ground-truth boxes from the instance PNGs and
writes standard one-based MOT box result rows. It evaluates boxes against
the MOTS annotations; it does not load the separate KITTI 2D tracking labels.
Prediction masks and the COCO codec are unnecessary for box evaluation.

For segmentation evaluation, install the COCO mask codec, retaining the extras
used by your environment:

```bash
uv sync --extra cpu --extra yolo --extra evolve --extra service --extra mots \
  --group dev --group test --group docs
```

Replace `cpu` with `cu130` on CUDA 13.0 hosts. Then add `--eval-masks` to the
evaluation command:

```bash
uv run --no-sync python -m boxmot.engine.cli eval \
  --dataset kitti-mots --split val --data-root datasets \
  --build runs/materializations/MASK_BUILD_ID \
  --tracker bytetrack --eval-masks
```

Use a build materialized with `--publish-masks`. With automatic preparation,
`--eval-masks` enables mask publication for you; the experiment must provide
masks through its detector or a segmentor. Mask evaluation rejects an explicit
build without published masks. Trackers that require masks still require them
in either evaluation mode. `tune` supports the same `--eval-masks` option.
Resuming tuning requires the original evaluation mode, so repeat `--eval-masks`
when resuming a segmentation run.
TrackEval is not required at runtime.

HOTA (including DetA and AssA), CLEAR (including MOTA, MOTP, and sMOTA),
Identity (including IDF1), and Count use the chosen IoU measure. Reports include car,
pedestrian, per-sequence scores, and detection- and class-averaged aggregates.
The standard CLEAR field names are retained; MOTA, MOTP, and sMOTA are computed
from mask matches with `--eval-masks`. Optional TrackEval J&F metrics and `--calibrate-kf` are not
supported for this dataset.

Ground truth comes directly from the uint16 instance PNGs. Segmentation ignore
handling follows TrackEval: only unmatched predictions with more than half
their mask area inside the ignore region are removed. Box mode applies the
same rule using box area overlapping the ignored pixels. `--fps` aligns predictions with the
selected original PNGs; `--sequence` restricts evaluation to chosen sequences.
The unannotated `test` split cannot be evaluated locally.

With `--eval-masks`, each sequence result is a space-separated MOTS file:

```text
frame_id track_id class_id height width compressed_coco_rle
```

Frames are zero-based. At native FPS, original frame numbers are preserved;
sampled builds use their recorded frame numbering. Classes are `1` (car) and
`2` (pedestrian). Native tracker masks take precedence. Box trackers retain
the masks of their matched detections; unmatched tracks without masks are
omitted. Overlapping pixels go to the higher-scoring track, with ties resolved
by the smaller track ID. Masks emptied by this operation are omitted.
Add `--show` or `--save` to visualize the same masks emitted for evaluation.

## Calibrate the KF at a selected FPS

Use `--fps 2 --calibrate-kf` to sample MOT17 at 2 FPS, fit Kalman noise from
the selected cached detections and ground truth, then evaluate once:

```bash
boxmot eval \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --tracker botsort \
  --fps 2 \
  --calibrate-kf
```

With no `--build`, evaluation prepares or reuses a compatible 2 FPS build.
With an explicit `--build`, `--fps` must match its recorded sampling rate;
omitting `--fps` uses that rate automatically. Detections, images, and ground
truth use the same selected frames, retaining their capture timestamps.

`--calibrate-kf` fits the five shared covariance scales directly; evaluation
then runs once. To calibrate before searching tracker parameters, use
[`boxmot tune --calibrate-kf`](../modes/tune.md#calibrate-the-kf-before-tracker-tuning)
with a materialized 2 FPS build. Calibration runs once before the trials,
which keep the fitted KF settings fixed.

Fixed-step prediction is the default. Add `--variable-dt` to calibrate and
predict using elapsed seconds from capture timestamps; `--fps` alone does
not enable it. Calibration requires ground truth and a supported Python
Kalman tracker. See [dataset FPS](../modes/eval.md#dataset-fps) and
[Kalman calibration](../modes/eval.md#kalman-calibration) for sampling details,
saved profiles, and evaluation on held-out sequences.

## Evaluate variable capture intervals

Use [`materialize --time-variant`](../modes/time-variant.md) to create a reproducible frame-loss
variant with original capture times, remapped ground truth, and a derived
perception build. After generating the MOT17-10-FRCNN example, evaluate its
`variable` split using the derived build ID printed by that command:

```bash
boxmot eval \
  --dataset datasets/mot/variants/mot17-10-frcnn-variable-time/dataset.yaml \
  --split variable \
  --build DERIVED_BUILD_ID \
  --tracker botsort \
  --variable-dt
```

Cached replay does not accept `--device`. Comparisons should use the same
variant and perception build, with tracker settings calibrated for the
selected timing mode. See [Kalman calibration](../modes/eval.md#kalman-calibration)
for fitting and reusing noise settings.

## Replay and postprocessing

Cached replay is sequence-parallel: each sequence is handled by a spawned
process with its own tracker instance. Automatic sizing uses the smaller of
the selected sequence count and the logical CPU count minus two, with at least
one worker. Use `--sequence-workers N` to set a positive integer cap instead;
the count stays bounded by the selected sequences. The evaluation UI shows a
separate frame-progress row for each sequence. Use `--sequence NAME` to limit a diagnostic run to one sequence;
repeat the option to select several. The parent reuses the frame counts already
validated during dataset setup and does not reopen perception payloads before
launch. Each worker reads detections for its assigned sequence and defers keyed
masks, embeddings, and image decoding until frame iteration, so optional
payloads are not replicated eagerly across the process pool.

Postprocessing such as interpolation applies to serialized tracker results,
not to the immutable build. Ground truth remains under the selected dataset
adapter rather than being embedded into the generic perception dataset.

Legacy `.npy`, `.npz`, and text-only perception caches are unsupported. They
are not read, migrated, or deleted.
