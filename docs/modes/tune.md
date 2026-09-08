# Tune

`tune` optimizes tracker parameters while replaying the same immutable
perception build for every trial. It requires an experiment and an explicit
build:

```bash
boxmot tune \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --build BUILD_ID \
  --tracker bytetrack \
  --n-trials 50
```

Materialize first if the build does not exist:

```bash
boxmot materialize --experiment mot17/ablation-yolox-lmbn.yaml
```

To tune at a lower dataset frame rate, materialize with `--fps 5` first and
select that build. Tuning reads its recorded FPS automatically; an explicit
`tune --fps` must match. Replayed images, detections, and ground truth share the
same selected frames and contiguous frame numbers, while capture timestamps
keep their original elapsed time. See [dataset FPS](eval.md#dataset-fps).

Tuning validates source, split, taxonomy, geometry, component fingerprints,
and the selected tracker's requirements once before optimization. Trials run no
detector, segmentor, or encoder and cannot silently select or create another
build. Worker count and retry policy are execution settings, not semantic
fingerprints.

The progress panel keeps HOTA, MOTA, and IDF1 visible for the best trial under
the configured objective and the latest completed trial, even as other trials
start or fail.

Use a native tracker with `--tracker-backend cpp` when that geometry and feature
combination is supported. Unsupported masks, per-class mode, or geometry are
rejected before a native trial starts.

## Calibrate the KF before tracker tuning

Add `--calibrate-kf` to estimate Kalman noise once from the build's cached
detections and ground truth, then tune the remaining tracker parameters.
For a 2 FPS run, first prepare a build at that rate:

```bash
boxmot materialize \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --fps 2
```

Use the printed build ID as `BUILD_ID` below. If you already have a compatible
2 FPS build, reuse its ID and skip materialization:

```bash
boxmot tune \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --build BUILD_ID \
  --tracker botsort \
  --fps 2 \
  --calibrate-kf \
  --n-trials 200
```

Calibration runs after build validation and before Ray and the search start.
The five calibrated covariance scales, their timing settings, and the filter's
reference process-noise priors stay fixed throughout all 200 tracker trials.
`adaptive_kf` also stays fixed for trackers that support it. No search-schema
edits are needed.

`--fps 2` checks the build's dataset sampling rate. It cannot resample a
full-rate build during tuning; use the 2 FPS build prepared above. You can
omit `--fps` to use the build's recorded rate automatically. Add
`--variable-dt` to calibrate and predict using capture timestamps in seconds.
Fixed-step mode remains the default.

The tuning directory contains `kf-tuning/calibrated.yaml` and
`kf-tuning/calibration.json`. The first is the calibrated starting tracker
configuration; the second records the data and estimates used for calibration.
The tuning result's `best.yaml` contains the selected tracker parameters with
the fixed KF settings included. Evaluate that result on held-out sequences.

To resume this search, use the same experiment and build with
`--resume-tune <tuning-directory>`. The saved calibration is restored
automatically. Combining `--calibrate-kf` with `--resume-tune` is rejected
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
