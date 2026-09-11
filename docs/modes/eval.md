# Evaluate

`eval` measures tracking performance from perception builds or saved sensor
datasets. For image trackers, it streams an immutable materialized build through
the live tracker API. Select an authored experiment with `--experiment` or use
dataset and component flags as shorthand for a matching catalog experiment.
In either case, `--build` is optional: when omitted, BoxMOT first materializes
(or reuses) a canonical build compatible with the selected tracker and then
evaluates it. Detector, segmentor, and appearance-encoder inference happens only
during that preparation step, never during replay. Automatic preparation
publishes image references, embeddings when the resolved tracker configuration
uses appearance, and masks when the selected tracker or scoring requires them.
Setting `use_embeddings: false` in `--tracker-config` skips the embedding stage,
as do motion-only trackers such as SFSORT. Appearance-enabled replay requires
cached embeddings; the live API's image-to-ReID fallback does not run during replay.

Matching complete builds are reused across BoxMOT releases and CPU, MPS, or
CUDA devices. Source data, weights, precision, preprocessing, class mapping,
stage settings, and requested outputs must still match. Reuse validates the
saved artifacts and keeps the original build ID without rerunning perception.

Preparation caches detector output separately from ReID embeddings. If a
compatible dataset, detector, geometry, and class mapping have already been
materialized, changing the ReID model skips detector inference and runs the
remaining derived stages. The cache lives below
`<selected-build-root>/.cache/detect`; with the default root, BoxMOT can also
import an identical v1 build or seed detections from another compatible v1
build in the former platform-cache location.
Detector-native masks or embeddings are not represented by this geometry cache,
so builds that require either output run their detector stage normally.

To select a catalog experiment through its components, provide `--dataset` and
`--detector`, plus the split and ReID profile when applicable:

```bash
boxmot eval \
  --dataset mot17 \
  --split ablation \
  --detector yolox-x-mot17 \
  --reid lmbn-n-duke \
  --tracker botsort
```

These flags resolve to the authored catalog experiment
`mot17/ablation-yolox-lmbn.yaml`. Its detector checkpoint, class map, and other
semantic settings remain authoritative. This is equivalent to:

```bash
boxmot eval \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --tracker botsort
```

Both forms materialize or reuse the exact same build. If the selectors match
no catalog experiment or more than one, evaluation reports an error. Matching
uses the exact dataset, detector, and ReID profiles plus the selected split.
The unique authored experiment supplies its detector checkpoint; append
`/CHECKPOINT` to `--detector` when the profile otherwise matches more than one
experiment. Omitting `--reid` selects only experiments without a ReID profile.
Use `--experiment` to select the intended configuration explicitly, or author
an experiment YAML for a combination absent from the catalog.

Pass exactly one of `--experiment` or `--dataset`. Direct component selectors
cannot be combined with `--experiment`.

`--device` selects the detector, segmentor, and ReID execution device when
automatic preparation needs new inference. It does not force regeneration of
matching saved outputs. It is rejected with an explicit `--build`, where no
perception model runs.

After materialization completes, evaluation consumes the exact path returned
by that build operation. It does not scan `--build-root`, select a latest
directory, or risk replaying another configuration's build.

Pass `--build` to reuse a specific build. Image-dataset evaluation requires
`--build` when no detector is selected, because the dataset config alone does
not select the perception components needed for materialization:

```bash
boxmot eval \
  --dataset mot17 \
  --split ablation \
  --build /srv/boxmot/materializations/BUILD_ID \
  --tracker bytetrack
```

An experiment selected by filename or component shorthand additionally fixes
semantic component fingerprints. Dataset mode uses the selected dataset
adapter for ground truth. Before tracking, evaluation verifies the build's
source catalog digest, split, class taxonomy, geometry, published requirements,
and—when applicable—component fingerprints.

If an explicitly selected build is missing or incompatible, evaluation fails
without modifying it or creating a replacement.

## Cache replay inputs for repeated runs

Add `--cache-inputs` to reuse the inputs consumed by evaluations or tuning runs:

```bash
boxmot eval \
  --dataset mot17 \
  --split ablation \
  --detector yolox-x-mot17 \
  --reid lmbn-n-duke \
  --tracker botsort \
  --cache-inputs
```

The first run validates the sources and prepares mapped arrays. Image workflows
cache AABB or OBB detections, requested embeddings and masks, and decoded image
pixels when the tracker or visualization needs them. Image references can point
to image files, NumPy arrays, or video frames. Ground-truth box annotations and
instance PNG labels also have reusable parsed caches.

The cache is tied to the selected source content, sequence, split, geometry,
and requested modalities. It is independent of the execution device and tracker
thresholds. Later runs can reuse it when switching between CPU, MPS, and CUDA.

With the default build layout, these files live under `runs/replay_cache/`.
For a custom build location, the cache directory sits beside the build-root
directory. They are derived data: the Parquet build remains authoritative and
is never rewritten. Incomplete or invalid derived entries are rebuilt from it.
You can remove the derived cache when no runs are using it to reclaim disk
space; `--cache-inputs` prepares it again when needed.

Saved sensor datasets support the same flag:

```bash
boxmot eval --dataset ./kitti-mots --tracker eagermot \
  --split val --cache-inputs
```

Sensor caches contain the frame timeline, declared 2D detections and packed masks,
3D detections, calibration, optional ego poses, and the ground truth selected for scoring. With
`--calibrate-kf`, calibration reuses the cached 3D annotations and observations.
RGB pixels are cached when visualization requests them; ordinary EagerMOT replay
only needs image dimensions. Masks remain packed on disk and are unpacked one
frame at a time. Sensor caches live under `<dataset-root>/.boxmot/replay_cache/`;
separate image-workflow annotation caches use `.boxmot/replay_cache/annotations/`
near their annotation sources.

The flag defaults to off because preparation takes time and extra disk space.
Use `--no-cache-inputs` to read the source formats directly. Source changes are
checked during preparation; an unchanged sensor study uses its prepared input
snapshot. Changed or incomplete caches are rebuilt. Tracker state, threshold
decisions, fusion, predictions, and metrics remain fresh for every run.

## Saved TrackR-CNN predictions

For downloaded KITTI TrackR-CNN text predictions, use
`boxmot track --tracker maf_hda` with `--detections`, `--images`, and `--instances`
directories. This command directly replays the saved boxes and
masks without materializing a perception build. See the
[MAF-HDA evaluation example](../trackers/maf_hda.md#evaluate-trackr-cnn-detections-on-kitti-mots)
for the full command.

## EagerMOT with saved sensor inputs

Evaluate a [multimodal sequence dataset](../config/datasets.md#multimodal-sequence-datasets)
through the same command:

```bash
boxmot eval \
  --dataset ./kitti-mots \
  --tracker eagermot \
  --split val
```

The dataset supplies sequence images, annotations, calibration, saved spatial
detections, and optional ego poses through `modalities` in its `dataset.yaml`.
Default mask evaluation also requires saved image detections with instance masks.
Each modality selects its encoding and relative paths; split overrides can
select different prediction sets.
You can pass the folder or its `dataset.yaml` file. Evaluation runs on CPU,
scores KITTI MOTS masks by default, and writes a new split directory under `runs/eagermot`.
Use `--project` to change that root or repeat `--sequence` to select sequences.
Sequences replay in parallel using the [automatic worker count](#sequence-parallelism).
Set `--sequence-workers 4` to allow at most four sequence workers.

To evaluate spatial tracks, declare `ground_truth_3d` with the existing native
KITTI tracking labels in `training/label_02`, then run:

```bash
boxmot eval --dataset ./kitti-mots --tracker eagermot \
  --split val --eval-3d --project runs/kitti-3d
```

The terminal shows **3D tracking HOTA/MOTA/IDF1** using volumetric box IoU.
Tracking scores are saved in `metrics.json/csv`, and predictions in
`kitti_3d/<sequence>.txt`. This custom tracking protocol uses all supplied target
GT; it does not apply official KITTI difficulty or DontCare filtering.

For additional official **2D/3D AP40 (Easy / Moderate / Hard)**, declare
[`ground_truth_objects`](../config/datasets.md#exact-object-labels-for-official-ap),
[install the official evaluators](../trackers/eagermot.md#evaluate-3d-tracks),
and add `--eval-ap` to the command. This writes `detection_metrics.json/csv`
and separately scores projected 2D tracking into `tracking_2d_metrics.json/csv`.
The main tracking scores remain volumetric 3D metrics.

Ground-truth masks are neither required nor loaded in this mode. `detections_2d`
can be omitted for tracking from 3D observations alone. Ego poses are optional;
without them, the tracker models motion in camera coordinates. Images still
provide the frame timeline and dimensions, and calibration remains required.
Declared tracking inputs, including any image predictions and ego poses, are consumed.
`--eval-3d` and `--eval-masks` are mutually exclusive; `--eval-3d` requires an
EagerMOT sensor dataset and is available only on `eval`. `--eval-ap` requires
`--eval-3d` and exact per-image object labels, including fractional truncation.
Only that additional option requires object annotations and the official evaluators.

For image and box trackers, select the [KITTI 2D dataset](../config/datasets.md#kitti-2d-tracking):

```bash
boxmot eval --dataset kitti-2d --tracker bytetrack --detector yolo26n \
  --split val --cache-inputs
```

It uses native `label_02` image boxes and TrackEval KITTI 2D HOTA/MOTA/IDF1.
Add `--reid osnet-x0-25-msmt17` when using BoT-SORT's appearance features.

For your own recordings, copy the
[sensor dataset template](../config/datasets.md#bring-your-own-sensor-dataset),
then supply synchronized images, calibration, 3D predictions, and ground truth
for the selected metric. Add ego poses when available and 2D mask predictions
for default mask evaluation. Custom sequence names and
splits are supported; detector outputs must follow the documented file formats.
The default evaluates car and pedestrian masks:

```bash
boxmot eval --dataset ./my-sensor-dataset --tracker eagermot \
  --split val --sequence drive-002
```

Load tuning's class profiles with `--class-config path/to/best.yaml`. Use
`--show` or `--save` to preview or record tracks, and add `--show-3d` to overlay
their estimated 3D cuboids. Saved videos go under the result directory's
`videos/` folder. `--class-config` and `--show-3d` apply only to this sensor workflow.
`--show` keeps sensor replay on the main thread, processing one sequence at a
time. With `--save` alone, each worker writes its sequence's video.

The shared Rich panel shows frame progress for each sequence and the selected
metrics. Add `--show-timing` to include replay timing in the result summary, or
`--verbose` to display tracker diagnostics alongside the panel.

With [3D annotations](../config/datasets.md#3d-ground-truth-for-kalman-calibration),
add `--calibrate-kf` to fit EagerMOT's five covariance scales per class before
evaluation. Reuse `<run>/kf-tuning/calibrated.yaml` with `--class-config`.
Calibration uses world coordinates when ego poses are declared, or camera
coordinates without them, matching tracking. Prediction advances one step per image;
see [3D Kalman calibration](../trackers/eagermot.md#calibrate-3d-kalman-noise).

Saved sensor evaluation reads predictions directly. Perception/build options
and TrackEval comparison are unavailable for these datasets;
an explicit `--device` must be `cpu`.

An incompatible selection reports a short reason and next step. The check uses
the selected split and registered [tracker inputs](../trackers/index.md#input-support):
declared tracking inputs marked `Unused` cause rejection. Ground truth is used
separately for scoring.

The direct saved-sensor workflow currently requires
`--tracker eagermot --tracker-backend python`. To intentionally evaluate an
image-only experiment, select only `images` and `ground_truth` in a separate
dataset config or explicit split override, then select a perception build or
detector through the ordinary evaluation workflow. Split modality overrides
set to `null` remove those inputs from the experiment.

Python callers use `boxmot.engine.eval.evaluator.run_eval(args)` and receive
the shared `ValidationResult`, including class-average metrics and `exp_dir`.
Set `args.eval_3d = True` to select spatial scoring.
See the [EagerMOT evaluation example](../trackers/eagermot.md#evaluate-downloaded-kitti-predictions).

## View tracking results

Add `--show` to preview annotated tracks, `--save` to write one MP4 per sequence,
or both. Replay uses the cached detections and embeddings, so no perception
models run with an explicit `--build`:

```bash
boxmot eval \
  --dataset mot17 \
  --split ablation \
  --build BUILD_ID \
  --tracker botsort \
  --tracker-config path/to/kf-tuning/calibrated.yaml \
  --show --save
```

Use the `calibrated.yaml` printed by a previous calibration run; omit
`--tracker-config` to use the tracker defaults. The saved configuration retains
its timing mode. `--show` and `--save` also work with `--calibrate-kf`: the tracker
replay is displayed or recorded after noise calibration finishes.

Preview follows source timestamps when available. Press **q** or **Esc** to
close the preview while evaluation continues. Annotated videos are written to
`<run>/videos/<sequence>.mp4` at 30 FPS. Frames are held across capture gaps,
quantized to a 30 FPS output grid, with one final 1/30-second frame. Sources
without timestamps use one output frame per input frame. `eval --fps` retains
its dataset-sampling meaning; it does not change the output video rate.

For image builds, visualization decodes source images and replays sequences
serially on the main thread, regardless of `--sequence-workers`. Reported replay
timing includes rendering, video writing, and preview pacing; omit these flags
for speed benchmarks.

## Dataset FPS

Use `--fps` to select a lower dataset frame rate:

```bash
boxmot eval \
  --dataset mot17 \
  --split ablation \
  --detector yolox-x-mot17 \
  --reid lmbn-n-duke \
  --tracker botsort --device mps --fps 5
```

For a 30 FPS sequence, `--fps 5` selects every sixth frame. Materialization,
replayed image loading, and ground truth share this selection. Tracker results
and ground truth use matching contiguous frame numbers, while capture
timestamps retain the original elapsed time. Fractional rates such as
`--fps 2.5` are supported. Selection uses the sequence's capture timeline,
including `timestamps.csv` when present; it never creates extra frames.

Automatic preparation first reuses an existing build at the requested rate.
Otherwise, it looks for a compatible published full-rate materialization and
copies the selected detections, embeddings, and masks into a smaller build
without loading the models. The parent must match the source images, capture
timestamps, ground-truth provenance, perception components, geometry, and class
mapping, and contain every requested payload. If no compatible parent exists,
materialization runs perception on the selected frames.

The target FPS enters the catalog and build identity, so each rate retains
its own immutable build. Omitting `--fps` during automatic preparation keeps
all original frames. With an explicit `--build`, omission uses the rate
recorded in that build. An explicit rate must match the build; omit `--build`
to prepare a different rate using automatic reuse, or run
`materialize --experiment YAML --fps RATE` first.

`--variable-dt` independently enables capture timestamps for Kalman prediction.
`track --fps` controls saved video playback speed; dataset frame selection
applies to `materialize`, `eval`, and `tune`.

Combine `--fps 2 --calibrate-kf` to fit Kalman noise using the selected 2 FPS
detections and ground truth before evaluating once. The
[calibration example](#kalman-calibration) also enables timestamp-based prediction.

## Legacy builds

An unbound build created before canonical experiment materialization can be
evaluated only with the explicit `--allow-noncanonical-build` escape hatch.
This relaxes the missing build-binding check; it does not turn incompatible
artifacts into compatible ones. Use it only after independently verifying the
build's source, split, detector, ReID model, geometry, and class taxonomy. For
example, to diagnose the known MMOT build on one sequence:

```bash
boxmot eval \
  --experiment mmot-obb/test-yolo11l-lmbn.yaml \
  --build faf16842d9d15fc048a50247df8ae927e7299d5760c9f461af4581cabfa279e6 \
  --data-root datasets/mot \
  --tracker botsort \
  --sequence data23-1 \
  --allow-noncanonical-build
```

Omit `--sequence data23-1` to evaluate every sequence. Canonical builds remain
the default and do not need this flag.

## Kalman calibration

EagerMOT supports [3D calibration](../trackers/eagermot.md#calibrate-3d-kalman-noise)
from saved sensor detections and 3D ground truth, using fixed frame steps.
The image-tracker workflow below supports AABB and OBB observations.

Use `--calibrate-kf` to estimate Kalman noise directly from cached detector
predictions and ground truth on the selected split, then evaluate the calibrated
tracker once:

```bash
boxmot eval \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --tracker botsort \
  --fps 2 \
  --variable-dt \
  --calibrate-kf
```

This example selects 2 FPS data and uses its capture timestamps for prediction.
Omit `--variable-dt` to calibrate in fixed-step mode, or omit `--fps` to retain
the original dataset rate during automatic preparation.

Perception is materialized or reused once. Calibration matches detections to
ground truth by class and IoU (at least `0.5`), then fits five dimensionless
covariance multipliers from the observed errors:

| Parameter | What provides the calibration residuals |
| --- | --- |
| `kf_process_position_scale` | Position-noise contribution to ground-truth box prediction errors |
| `kf_process_velocity_scale` | Velocity-noise contribution to the same prediction errors |
| `kf_measurement_noise_scale` | Matched detector boxes minus ground-truth boxes |
| `kf_initial_position_scale` | Detection errors at each GT object's first matched observation |
| `kf_initial_velocity_scale` | Zero-initialized velocity errors relative to local ground-truth motion |

Measurement and initialization scales use residual second moments. Both process
scales are fitted together from constant-velocity prediction errors across
three consecutive annotated frames. The fit accounts for process noise from
both intervening intervals and the covariance between adjacent prediction
errors. Adjacent error pairs require four consecutive annotated frames and
help separate position from velocity noise, including at regular frame rates.
Ground-truth motion contributes even when a detection is missed; missing
annotations break the motion samples.
The first matched detection for each object supplies an initialization proxy.
Calibration requires at least one valid detection/ground-truth match. If a
scale lacks sufficient evidence, its current value is retained and the report
records why.

The estimates scale the selected filter's reference covariance priors. Values
multiply covariance, not standard deviation. Calibration keeps all other
tracker settings fixed, including the timing mode. The filter's internal
base priors remain fixed; the five shared scales provide the noise-calibration
interface.

`eval --calibrate-kf` does not launch Ray Tune or Optuna, replay HOTA trials, or
require the `evolve` extra. Use [`boxmot tune`](tune.md#kalman-noise-and-timing)
for metric-based search over tracker parameters; Kalman settings remain fixed
during that search. Covariance multipliers are runtime settings estimated by
calibration, with defaults retained in the tracker YAML and no search ranges.
Use [`tune --calibrate-kf`](tune.md#calibrate-the-kf-before-tracker-tuning) to
calibrate once before searching the other tracker settings, keeping the KF
calibration fixed throughout the search.
A positive numerical floor keeps calibrated covariances valid. See
[OC-SORT base process noise](tune.md#oc-sort-base-process-noise) for the
relationship between model priors and calibrated multipliers.

Calibration supports AABB and OBB ground truth with Python ByteTrack, BotSort,
StrongSort, OcSort, DeepOcSort, HybridSort, BoostTrack, and OccluBoost. Native
backends and trackers without a Kalman filter are rejected before automatic
materialization. HybridSORT's confidence state has no ground-truth confidence
target, so confidence residuals are excluded from fitting; the shared scales
still apply to that state during tracking.

### Timing and saved settings

Fixed-step prediction remains the default. Add `--variable-dt` to fit and use
noise in seconds, with elapsed intervals from capture timestamps. Neither
mode searches `dt`, and measurement noise is not scaled by elapsed time.
`--fps 2` selects dataset frames at 2 FPS; it does not enable `--variable-dt`.

`kf_reference_dt_s` fixes the interval used to convert historic per-frame priors
into seconds: its default `0.03333333333333333` represents a 30 FPS reference.
It is not the source clock or the prediction interval. This reference and
`kf_time_unit` stay fixed during calibration and tracker tuning. See
[time-unit conversion](../python/index.md#elapsed-time) for the covariance
scaling rules.

Each run writes:

- `<run>/kf-tuning/calibrated.yaml`: resolved scalar tracker settings, tracker
  name, `variable_dt`, explicit `kf_time_unit` (`frames` or `seconds`), and
  `kf_reference_dt_s`.
- `<run>/kf-tuning/calibration.json`: calibration evidence, timing settings,
  and the final evaluation result.

Calibration fits noise statistics, so it does not guarantee a higher HOTA.
The final replay uses the fitting split. Evaluate the saved settings on
separate sequences before judging how well they generalize. Initial velocity
uses a local ground-truth finite difference as a proxy. Annotation noise and
camera motion contribute to motion errors measured in image coordinates.
Measurement estimates describe
detections that pass the IoU match threshold; they do not model false positives.

Use `--tracker-config` to evaluate the saved configuration on a separate split
with ground truth and a compatible build. For example, after preparing a
validation build whose sequences are held out from calibration:

```bash
boxmot eval \
  --dataset mot17 \
  --split val \
  --build /srv/boxmot/materializations/HELD_OUT_BUILD \
  --tracker botsort \
  --tracker-config path/to/kf-tuning/calibrated.yaml
```

`--tracker-config` also accepts a partial scalar YAML or a built-in preset such
as `botsort-mot17-ablation`. It overlays tracker defaults. Explicit runtime
flags such as `--asso-func` can override scalar parameters, but a calibrated
config's time units must match the selected mode. For example,
`--fixed-dt --tracker-config seconds-config.yaml` is rejected when that file
declares `kf_time_unit: seconds`. Recalibrate in the intended mode instead of
reinterpreting the saved values. Keep the detector, geometry, backend, class
selection, and per-class tracking behavior consistent with calibration.

### Live streams

Direct calibration needs ground truth. Run it on representative timestamped
recordings, including dropped frames and missed detections, then load the
saved profile for live tracking. Keep the same detector, geometry, timing mode,
and reference interval:

```bash
boxmot track \
  --source <stream> \
  --tracker botsort \
  --tracker-config path/to/kf-tuning/calibrated.yaml
```

BoostTrack and OccluBoost also support experimental online process-noise
adaptation with `adaptive_kf: true`. This learns from each track's prediction
errors; it does not learn measurement noise or initial uncertainty and does
not measure tracking accuracy. Incorrect associations can distort its estimates.
Calibration estimates the starting priors and preserves this setting; it does
not fit the online adaptation behavior. Keep it consistent during evaluation
and deployment.
Other trackers support the calibrated static profile and interval-aware
prediction; `--variable-dt` itself does not enable online noise adaptation.

Live intervals must come from trustworthy source timestamps. A nominal-FPS
fallback cannot reveal unseen capture dropouts or reconnect duration; use
capture/PTS metadata or provide `Frame.timestamp_s` through the Python API.
Elapsed inference time is not a capture timestamp.

## Sequence parallelism

Parallel evaluation gives each sequence its own tracker in an isolated
process. Tracker state and native handles never cross sequence or process
boundaries.
By default, the worker count is the smaller of the selected sequence count and
the logical CPU count minus two, with at least one worker when sequences are
selected: `min(sequences, max(1, logical_cpus - 2))`. For example, nine selected
sequences on a machine with eight logical CPUs use six workers.

`--sequence-workers N` overrides automatic sizing with a positive integer cap;
the active worker count never exceeds the number of selected sequences.
Use `--sequence-workers 1` to process sequences one at a time. The Rich panel
reports frame progress separately for every sequence while those jobs run.

Image-build previews and saved videos use serial replay. EagerMOT sensor
replay runs serially with `--show`; `--save` alone supports parallel video
writing.

Use `--sequence` to replay only one sequence while diagnosing a run. Repeat
the option to select more than one sequence:

```bash
boxmot eval \
  --experiment mmot-obb/test-yolo11l-lmbn.yaml \
  --build BUILD_ID \
  --sequence data23-1 \
  --tracker botsort
```

`boxmot eval` reuses the sequence frame counts already validated during dataset
setup. A standalone replay call falls back to reading only the samples table's
sequence IDs. Neither path eagerly loads masks or embeddings in the
coordinator. It marks each sequence as queued before starting the process pool.
A worker then opens only its assigned sequence and streams optional masks and
embeddings in bounded Arrow batches as frame iteration advances. Images are
decoded when the resolved tracker or `--show`/`--save` requires source pixels. A
dimensions-only tracker such as SFSORT receives width and height from cached
sample metadata without opening the source image unless visualization is enabled.

Current materializations also use bounded Parquet row groups so workers can
prune unrelated `sample_id` ranges. Older `boxmot.dataset/v1` builds remain
readable, but builds created with large row groups can require more I/O during
the initial worker-loading phase.

## Build resolution

- When `--build` is omitted with `--experiment` or `--dataset` plus
  `--detector`, BoxMOT resolves the authored experiment and materializes its
  deterministic build below `--build-root`; an identical complete build is
  validated and reused.
- An existing `--build` path is used directly.
- A build ID is looked up only below `--build-root`.
- `--build-root` defaults to `BOXMOT_BUILDS_DIR`, then
  `./runs/materializations`.
- Dataset-only evaluation requires `--build` when no detector is selected.
- There is no latest-build selection.

The selected raw data root is used to verify ground-truth provenance. It
defaults to `./datasets/mot`; pass `--data-root` explicitly to use another
location. When the selected split is absent, its configured Hugging Face
`per_split` resource is downloaded before materialization or ground-truth
validation. Existing populated splits are reused without downloading;
archive-backed datasets still require explicit setup.

Repeated evaluations reuse hashes and image dimensions only when a source or
model file's path, device, inode, mode, size, modification time, and change
time are unchanged. This shared metadata lives in the platform cache;
the source tree is still traversed on every run so added and removed files are
detected. This metadata cache is separate from the content-addressed detector
output cache; materialization still establishes the source and model content
digests before selecting either a complete build or reusable detections.

## Output

Tracker output is serialized to MOT text only at the evaluation boundary.
Reusable postprocessors can consume those files separately and never rewrite
the immutable perception build. `--compare-trackeval` is available for
supported AABB MOTChallenge datasets.

## Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.eval
    :command: eval
    :prog_name: boxmot eval
    :depth: 0
