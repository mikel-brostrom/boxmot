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

Use a native tracker with `--tracker-backend cpp` when that geometry and feature
combination is supported. Unsupported masks, per-class mode, or geometry are
rejected before a native trial starts.

## Kalman noise and timing

For Python Kalman trackers, joint tuning includes five covariance multipliers:
position and velocity process noise, measurement noise, and initial position
and velocity uncertainty. Each defaults to `1.0` and uses a logarithmic
`0.01–100` range. The tracker YAML is the common source for these ranges and
the [KF-only HOTA search](eval.md#kalman-calibration) activated by
`eval --kf-tuning`.

To continue from a KF-only calibration, add
`--tracker-config path/to/kf-tuning/best.yaml` to the `tune` command. This
loads the saved tracker settings as the baseline for joint optimization,
including their timing mode, units, and reference interval. The selector also
accepts scalar runtime YAMLs and built-in presets.

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

### Fix OC-SORT base process noise

For Python OC-SORT and DeepOCSORT, `Q_xy_scaling` sets the base process noise
for centre velocity, while `Q_s_scaling` sets it for bounding-box area velocity.
The shared velocity multiplier scales both:

```text
centre-velocity noise = Q_xy_scaling × kf_process_velocity_scale
area-velocity noise   = Q_s_scaling  × kf_process_velocity_scale
```

These products describe the reference noise before time-unit conversion and
integration into `Q(dt)`. Searching all three parameters allows different
combinations to produce identical noise. Their relative centre/area balance
remains a distinct choice; the five shared multipliers cannot change it.
In OBB mode, these trackers also tie angular-velocity base noise to `Q_s_scaling`.

To preserve the existing base priors while searching only the five shared
Kalman multipliers, replace the two entries in the selected built-in search
schema, `boxmot/configs/trackers/ocsort.yaml` or
`boxmot/configs/trackers/deepocsort.yaml`, with:

```yaml
Q_xy_scaling:
  default: 0.01

Q_s_scaling:
  default: 0.0001
```

Remove their `type` and `range` fields, and keep the five `kf_*_scale` search
entries. Joint tuning holds default-only entries fixed while continuing to
search the remaining tracker and Kalman parameters. This retains the existing
noise balance and removes the two extra search dimensions.
Start a new tuning run after changing the schema; `--resume-tune` restores
the previous search state.

`eval --kf-tuning` already holds these base settings fixed, so it needs no
schema change. A scalar `--tracker-config` can override their resolved values;
freezing a parameter during joint tuning requires the default-only search
entry above. The timing settings remain fixed in either case.

## Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.tune
    :command: tune
    :prog_name: boxmot tune
    :depth: 0
