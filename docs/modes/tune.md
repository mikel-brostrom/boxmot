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

## EagerMOT with saved KITTI sensor inputs

Pass a [KITTI fusion dataset](../config/datasets.md#kitti-fusion-datasets) to
`--dataset` with `--tracker eagermot`:

```bash
boxmot tune \
  --dataset ./kitti-mots \
  --tracker eagermot \
  --n-trials 50 \
  --seed 0
```

The folder's `dataset.yaml` defines sequence locations, classes, splits, and
the replay configuration. Each sequence contains images, ground truth,
calibration, and ego poses. `replay.yaml` selects the saved 2D/3D prediction
sets for each split. You can also pass `dataset.yaml` itself; an absolute
`--dataset` path works from any working directory. Keep the `./` prefix when
selecting the local folder by name; bare `kitti-mots` selects the built-in
image dataset profile.

With the `mots` and `evolve` extras installed, this runs serial Optuna trials
on CPU, replaying independent sequences in parallel within each trial, and
maximizes class-average mask HOTA across car and pedestrian profiles.
The [automatic worker count](eval.md#sequence-parallelism) uses the selected
sequences and logical CPU count; `--sequence-workers 4` caps it at four workers
per trial. The first trial uses the default KITTI profiles. Add
`--sequence 0002` to select one validation sequence, repeat `--sequence` for
several, or use `--split train` or `--split fulltrain` to select another
dataset split. `--project` defaults to `runs/eagermot-tune`; each run saves
`best.yaml` for `boxmot eval --tracker eagermot --class-config`.

The shared Rich tuning panel shows trial and sequence progress, the best HOTA,
and the saved profile paths. Add `--verbose` to display tracker and Optuna logs.

This sensor workflow reads the dataset and selected predictions directly.
It supports `--search-alg optuna`; an explicit device must be `cpu`.
`--max-concurrent-trials` accepts `0` (default) or `1`, keeping trials serial.
Objective selectors must use `HOTA`. Perception and build options,
`--calibrate-kf`, and `--resume-tune` are unavailable for fusion datasets.
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

Each trial uses the [automatic sequence worker count](eval.md#sequence-parallelism)
unless `--sequence-workers` supplies a positive integer cap. For example,
`--sequence-workers 4` allows up to four sequence worker processes per trial,
bounded by the number of selected sequences. Use `--max-concurrent-trials` to
limit how many image-tracker trials run at once; sequence workers are allocated
separately to each trial.

The progress panel keeps HOTA, MOTA, and IDF1 visible for the best trial under
the configured objective and the latest completed trial, even as other trials
start or fail.

Use a native tracker with `--tracker-backend cpp` when that geometry and feature
combination is supported. Unsupported masks, per-class mode, or geometry are
rejected before a native trial starts.

## Calibrate the KF before tracker tuning

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

To reuse an existing calibration without fitting again, start a new tuning
run with `--tracker-config path/to/kf-tuning/calibrated.yaml` and omit
`--calibrate-kf`. The loaded KF values stay fixed while the other tracker
parameters are optimized. Without a profile, the built-in KF defaults stay
fixed. The selector also accepts partial scalar runtime YAMLs and built-in
presets.

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
