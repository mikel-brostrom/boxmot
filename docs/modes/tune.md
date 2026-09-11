# Tune

`tune` optimizes tracker parameters on a selected dataset. For image trackers,
it prepares or reuses a canonical perception build once, optionally calibrates
the Kalman filter, and replays that same immutable build in every trial.
Prepare Ray tuning dependencies with `boxmot install --extra evolve`; see
[Install dependencies](install.md).

```bash
boxmot tune \
  --dataset mot17 \
  --split ablation \
  --detector yolox-x-mot17 \
  --reid lmbn-n-duke \
  --tracker botsort \
  --device mps \
  --fps 2 \
  --calibrate-kf \
  --n-trials 200
```

This command prepares 2 FPS data, calibrates the KF from cached detections and
ground truth, then runs 200 tracker trials with the KF settings fixed. Omit
`--calibrate-kf` to keep default or loaded KF settings without fitting them.

The dataset and component selectors resolve a matching authored experiment.
You can select it directly with `--experiment mot17/ablation-yolox-lmbn.yaml`
in place of `--dataset`, `--detector`, and `--reid`. Missing or ambiguous
catalog matches produce an error; use `--experiment` to select the intended
configuration. See [experiment workflows](../guides/experiments.md).

## KITTI with image trackers

The [KITTI 2D config](../config/datasets.md#kitti-2d-tracking) uses native
tracking box labels with the standard materialization and tuning workflow:

```bash
boxmot tune --dataset kitti-2d --tracker bytetrack --detector yolo26n \
  --split train --n-trials 50 --cache-inputs
```

Install `--extra trackeval` alongside the detector and tuning extras. Use
`--calibrate-kf` to fit 2D filter noise before searching tracking parameters.
BoT-SORT can use the supplied OSNet experiments by adding
`--reid osnet-x0-25-msmt17` and selecting `--tracker botsort`.

## EagerMOT with saved sensor inputs

Pass a [multimodal sequence dataset](../config/datasets.md#multimodal-sequence-datasets) to
`--dataset` with `--tracker eagermot`:

```bash
boxmot tune \
  --dataset ./kitti-mots \
  --tracker eagermot \
  --n-trials 50 \
  --seed 0
```

The folder's `dataset.yaml` defines classes, splits, and `modalities` with
encodings and paths for images, ground truth, calibration, ego poses, and
saved 2D/3D detections. Split overrides can select different prediction sets.
You can also pass `dataset.yaml` itself; an absolute
`--dataset` path works from any working directory. Keep the `./` prefix when
selecting the local folder by name; bare `kitti-mots` selects the built-in
image dataset profile.

For your own synchronized camera and 3D observations, start from the
[sensor dataset template](../config/datasets.md#bring-your-own-sensor-dataset).
It supports custom split, partition, and sequence names with the documented
calibration, pose, image, and prediction formats. Supply car/pedestrian
ground-truth masks and image prediction masks for the tuning objective:

```bash
boxmot tune --dataset ./my-sensor-dataset --tracker eagermot \
  --split train --n-trials 50 --seed 0
boxmot eval --dataset ./my-sensor-dataset --tracker eagermot \
  --split val --class-config runs/eagermot-tune/train/best.yaml
```

Use the actual `best.yaml` path printed by tuning. The template keeps fitting
and validation sequences separate; record each detector's model, checkpoint,
and training data in your dataset README or YAML comments.

With the `mots` and `evolve` extras installed, this runs serial Optuna trials
on CPU, replaying independent sequences in parallel within each trial, and
maximizes class-average mask HOTA across car and pedestrian profiles.
The [automatic worker count](eval.md#sequence-parallelism) uses the selected
sequences and logical CPU count; `--sequence-workers 4` caps it at four workers
per trial. The first trial uses the starting class profiles, including any
loaded or calibrated settings. Add
`--sequence 0002` to select one validation sequence, repeat `--sequence` for
several, or use `--split train` or `--split fulltrain` to select another
dataset split. `--project` defaults to `runs/eagermot-tune`; each run saves
`best.yaml` for `boxmot eval --tracker eagermot --class-config`.

The shared Rich tuning panel shows trial and sequence progress, the best HOTA,
and the saved profile paths. Add `--verbose` to display tracker and Optuna logs.

This sensor workflow reads the dataset and selected predictions directly.
It supports `--search-alg optuna`; an explicit device must be `cpu`.
`--max-concurrent-trials` accepts `0` (default) or `1`, keeping trials serial.
Objective selectors must use `HOTA`. Perception and build options and
`--resume-tune` are unavailable for fusion datasets.

Supply [3D annotations](../config/datasets.md#3d-ground-truth-for-kalman-calibration)
and add `--calibrate-kf` to fit the 3D Kalman noise once before Optuna starts:

```bash
boxmot tune --dataset ./my-sensor-dataset --tracker eagermot \
  --split train --calibrate-kf --n-trials 50 --seed 0
```

The five covariance scales and `is_angular` stay fixed for each class during
the search. Reuse calibration without fitting again with
`--class-config path/to/kf-tuning/calibrated.yaml`. Both that starting profile
and the final `best.yaml` contain separate `car` and `pedestrian` settings.
`--class-config` also holds the loaded `is_angular` choices fixed; tuning
without either flag can search those choices.
Ego poses remain fixed; EagerMOT advances one frame per image. See
[3D Kalman calibration](../trackers/eagermot.md#calibrate-3d-kalman-noise).

An incompatible selection reports a short reason and next step. The check uses
the selected split and registered [tracker inputs](../trackers/index.md#input-support):
declared tracking inputs marked `Unused` cause rejection. Ground truth is used
separately for the tuning objective.

Direct saved-sensor tuning currently requires
`--tracker eagermot --tracker-backend python`. To intentionally tune an image-only
experiment, select only `images` and `ground_truth` in a separate dataset config
or explicit split override, plus a perception build or detector. Split modality
overrides set to `null` remove those inputs from the experiment.

See the [EagerMOT tuning example](../trackers/eagermot.md#tune-separate-class-profiles)
for inputs and outputs.

Python callers use `boxmot.engine.tuning.tuner.run_tune(args)` for both image
builds and sensor datasets. It returns a `TuneResult` with the completed trials,
best metrics, and `best_yaml` path. For EagerMOT, `best_config` contains separate
`car` and `pedestrian` profiles, matching the exported YAML.

## Build preparation and reuse

For image trackers, when `--build` is omitted, tuning resolves the canonical
build from the selected experiment, source data, perception settings, and
requested frame rate. A matching complete build is validated and reused;
otherwise, materialization runs once before calibration and Ray start. Reuse requires
matching source and semantic component fingerprints, geometry, class taxonomy,
and the payloads needed by the tracker. It never selects a latest build.

`--device` controls perception during automatic preparation, for example
`mps`, `cuda:0`, or `cpu`. With an explicit `--build`, no perception models run
and `--device` is rejected. Use either an experiment or a dataset to replay a
specific build:

```bash
boxmot tune \
  --dataset mot17 \
  --split ablation \
  --build BUILD_ID \
  --tracker botsort \
  --n-trials 200
```

A build ID resolves below `--build-root`; an existing build path is used
directly. An explicitly selected build must already exist and be compatible.
Dataset-only selection requires `--build`; automatic preparation needs an
experiment or `--dataset` plus `--detector` and optional `--reid`.

With automatic preparation, `--fps 2` selects a 2 FPS build. A compatible
full-rate cache can supply the selected detections and embeddings without
rerunning perception. With an explicit `--build`, omitting `--fps` uses its
recorded rate, and an explicit rate must match it. Replayed images,
detections, and ground truth share the selected frames and contiguous frame
numbers, while capture timestamps retain their original elapsed time. See
[dataset FPS](eval.md#dataset-fps).

Tuning validates the build and the tracker's requirements before optimization.
Trials run no detector, segmentor, or encoder and cannot select or create
another build. Worker count and retry policy are execution settings, not
semantic fingerprints.

Automatic preparation caches embeddings when the baseline uses them or the
search can enable `use_embeddings`. A scalar `--tracker-config` sets the
baseline; searchable parameters can still change in trials. To omit embeddings
throughout tuning, both the baseline and search must keep appearance disabled.
Appearance-enabled trials require cached embeddings rather than live ReID inference.

Each trial uses the [automatic sequence worker count](eval.md#sequence-parallelism)
unless `--sequence-workers` supplies a positive integer cap. For example,
`--sequence-workers 4` allows up to four sequence worker processes per trial,
bounded by the number of selected sequences. Use `--max-concurrent-trials` to
limit how many image-tracker trials run at once; sequence workers are allocated
separately to each trial.

Each reusable image tuning actor and each sensor study keeps its sequence-worker
pool across trials. Every trial still creates fresh
trackers, pipelines, frame cursors, and output files. Worker pools are closed
when tuning ends; failed worker operations discard the pool before reuse.
This applies to image search backends and the sensor Optuna workflow without an
additional flag.

Add `--cache-inputs` to also reuse mapped inputs on disk across tuning sessions
and evaluation commands:

```bash
boxmot tune \
  --dataset mot17 \
  --split ablation \
  --build BUILD_ID \
  --tracker botsort \
  --n-trials 200 \
  --cache-inputs
```

The same flag works with multimodal datasets and per-class KF calibration:

```bash
boxmot tune --dataset ./kitti-mots --tracker eagermot \
  --split train --calibrate-kf --cache-inputs --n-trials 50
```

This requires the [3D ground-truth declaration](../config/datasets.md#3d-ground-truth-for-kalman-calibration)
used by calibration. Omit `--calibrate-kf` when tuning without 3D annotations.
Sensor caches reuse parsed 2D/3D detections, packed masks, calibration, ego poses,
and ground truth. Image builds cache requested detections, embeddings, masks,
and image pixels. Filtering remains inside each trial, so differing detection
thresholds and association settings reuse the same unfiltered input cache.
See [replay input caching](eval.md#cache-replay-inputs-for-repeated-runs) for
storage, preparation, and cleanup details.

The progress panel keeps HOTA, MOTA, and IDF1 visible for the best trial under
the configured objective and the latest completed trial, even as other trials
start or fail.

Use a native tracker with `--tracker-backend cpp` when that geometry and feature
combination is supported. Unsupported masks, per-class mode, or geometry are
rejected before a native trial starts.

## Calibrate the KF before tracker tuning

For saved sensor datasets, use the [EagerMOT workflow above](#eagermot-with-saved-sensor-inputs).
The build preparation and resume behavior below apply to image trackers.

Add `--calibrate-kf`, as in the first example, to estimate Kalman noise once
from the prepared build's cached detections and ground truth, then tune the
remaining tracker parameters. The flag works with automatic preparation or
an explicitly selected build.

Calibration runs after build validation and before Ray and the search start.
The five calibrated covariance scales, their timing settings, and the filter's
reference process-noise priors stay fixed throughout all 200 tracker trials.
`adaptive_kf` also stays fixed for trackers that support it. No search-schema
edits are needed.

Add `--variable-dt` to calibrate and predict using capture timestamps in
seconds. Fixed-step mode remains the default; `--fps` does not change the
Kalman timing mode.

The tuning directory contains `kf-tuning/calibrated.yaml` and
`kf-tuning/calibration.json`. The first is the calibrated starting tracker
configuration; the second records the data and estimates used for calibration.
The tuning result's `best.yaml` contains the selected tracker parameters with
the fixed KF settings included. Evaluate that result on held-out sequences.

Resume with `--resume-tune <tuning-directory>`, using the same experiment or
dataset selection and build. The saved calibration is restored automatically.
Combining `--calibrate-kf` with `--resume-tune` is rejected
because a fresh calibration would change the existing search.
Resuming retains the original search space; start a new run to use updated
search definitions, including the removal of KF search dimensions.

`eval --calibrate-kf` instead calibrates and evaluates the tracker once. See
[Kalman calibration](eval.md#kalman-calibration) for the estimator, supported
trackers, and limitations.

## Kalman noise and timing

New tracker tuning runs hold Kalman settings fixed. The five covariance
multipliers retain their runtime defaults of `1.0` in the tracker YAML;
`adaptive_kf` and timing settings also retain their YAML runtime defaults where
supported. OC-SORT's base process-noise priors are fixed inside the filter.
These settings have no search ranges.
Use [Kalman calibration](eval.md#kalman-calibration) to estimate covariance
scales from detections and ground truth.

For image trackers, reuse an existing calibration by starting a new tuning run
with `--tracker-config path/to/kf-tuning/calibrated.yaml` and omit
`--calibrate-kf`. The loaded KF values stay fixed while the other tracker
parameters are optimized. Without a profile, the built-in KF defaults stay
fixed. The selector also accepts partial scalar runtime YAMLs and built-in
presets.

EagerMOT uses `--class-config` for its separate car and pedestrian profiles.
Its 3D filter supports covariance calibration in fixed-step mode only.

`variable_dt`, `kf_time_unit`, and `kf_reference_dt_s` are fixed runtime
settings. They are not tuning parameters, and elapsed `dt` is never sampled.
Use `--variable-dt` to select elapsed-seconds prediction explicitly, or keep
the default fixed-step mode. The reference interval defaults to `1/30` second
and describes the original noise priors, not the source timestamps.

Saved configurations record the timing mode, explicit units, and reference
interval. Reuse them with `--tracker-config` in `track`, `eval`, or another
`tune` run in the same mode. A conflicting time-unit override is rejected. Tune results
measure the fitting split; evaluate on separate held-out sequences before
judging whether the selected settings improve deployment accuracy.

### OC-SORT base process noise

OC-SORT and DeepOCSORT use fixed reference process covariances of `0.01` for
centre velocity and `0.0001` for bounding-box area velocity. OBB angular
velocity also uses `0.0001`. Python Kalman calibration scales these priors
with the shared velocity multiplier:

```text
centre-velocity noise = 0.01   × kf_process_velocity_scale
area-velocity noise   = 0.0001 × kf_process_velocity_scale
angular-velocity noise = 0.0001 × kf_process_velocity_scale  (OBB)
```

These products describe reference noise before time-unit conversion and
integration into `Q(dt)`. The relative balance between centre, area, and
angular velocity noise is fixed by the filter; calibration scales them
together. The five shared `kf_*_scale` settings are the noise-calibration
interface.

## Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.tune
    :command: tune
    :prog_name: boxmot tune
    :depth: 0
