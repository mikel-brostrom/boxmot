# Evaluate

`eval` measures a tracker by streaming an immutable materialized build through
the live tracker API. Select an authored experiment with `--experiment` or use
dataset and component flags as shorthand for a matching catalog experiment.
In either case, `--build` is optional: when omitted, BoxMOT first materializes
(or reuses) a canonical build compatible with the selected tracker and then
evaluates it. Detector, segmentor, and appearance-encoder inference happens only
during that preparation step, never during replay. Automatic preparation
publishes image references, embeddings for appearance-capable trackers, and
masks when the selected tracker requires them. Motion-only trackers such as
SFSORT skip the embedding stage entirely.

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

`--device` selects the detector, segmentor, and ReID execution device for this
automatic preparation. It is rejected with an explicit `--build`, where no
perception model runs.

After materialization completes, evaluation consumes the exact path returned
by that build operation. It does not scan `--build-root`, select a latest
directory, or risk replaying another configuration's build.

Pass `--build` to reuse a specific build. Dataset-only evaluation requires
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
  --tracker-config path/to/kf-tuning/best.yaml \
  --show --save
```

Use the `best.yaml` printed by a previous calibration run; omit
`--tracker-config` to use the tracker defaults. The saved configuration retains
its timing mode. `--show` and `--save` also work with `--kf-tuning`: only the
final replay of the selected configuration is displayed or recorded, not the
search trials.

Preview follows source timestamps when available. Press **q** or **Esc** to
close the preview while evaluation continues. Annotated videos are written to
`<run>/videos/<sequence>.mp4` at 30 FPS. Frames are held across capture gaps,
quantized to a 30 FPS output grid, with one final 1/30-second frame. Sources
without timestamps use one output frame per input frame. `eval --fps` retains
its dataset-sampling meaning; it does not change the output video rate.

Visualization decodes source images and replays sequences serially on the main
thread, regardless of `--n-threads`. Reported replay timing includes rendering,
video writing, and preview pacing; omit these flags for speed benchmarks.

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

Use `--kf-tuning` to calibrate five Kalman covariance multipliers for HOTA on
the selected split, then run evaluation with the selected configuration:

```bash
boxmot eval \
  --experiment mot17/ablation-yolox-lmbn.yaml \
  --tracker botsort \
  --variable-dt \
  --kf-tuning \
  --kf-trials 20
```

The default is 20 trials total, including the starting configuration. Optuna
uses seed 0. `--kf-trials 1` evaluates only the starting configuration, and
`--kf-trials` requires `--kf-tuning`. Perception data is materialized or reused
once; all trials replay the same build.
This requires the `evolve` installation extra, which provides Optuna.

| Parameter | What it scales | Default | Search range |
| --- | --- | --- | --- |
| `kf_process_position_scale` | Position process noise | `1.0` | `0.01–100` |
| `kf_process_velocity_scale` | Velocity process noise | `1.0` | `0.01–100` |
| `kf_measurement_noise_scale` | Measurement covariance | `1.0` | `0.01–100` |
| `kf_initial_position_scale` | Initial position covariance | `1.0` | `0.01–100` |
| `kf_initial_velocity_scale` | Initial velocity covariance | `1.0` | `0.01–100` |

These logarithmic search ranges come from the tracker YAML and are shared with
[joint tracker tuning](tune.md). They are dimensionless multipliers of the
filter's reference priors, after conversion into the selected time units. The
same ranges apply in fixed-step and elapsed-seconds modes.

For OC-SORT and DeepOCSORT, this search holds `Q_xy_scaling` and `Q_s_scaling`
fixed. To keep them fixed during joint tuning too, see
[fixing base process noise](tune.md#fix-oc-sort-base-process-noise).

These values multiply covariance, not standard deviation. The search keeps
other tracker settings fixed, including `variable_dt`; timestamps do not
enable variable timing. Use `--variable-dt` explicitly to calibrate the
experimental elapsed-seconds mode. Calibration supports Python ByteTrack,
BotSort, StrongSort, OcSort, DeepOcSort, HybridSort, BoostTrack, and OccluBoost.
Native backends, SFSORT, and SAM2 are rejected before automatic materialization.

`kf_reference_dt_s` fixes the interval used to convert historic per-frame priors
into seconds: its default `0.03333333333333333` represents a 30 FPS reference.
It is not the source clock or the prediction interval. Actual elapsed intervals
come from capture timestamps. This reference and `kf_time_unit` are default-only
runtime settings, never search dimensions. See [time-unit conversion](../python/index.md#elapsed-time)
for the covariance scaling rules.

Each run writes `<run>/kf-tuning/best.yaml`, containing the resolved scalar
tracker parameters, tracker name, `variable_dt`, explicit `kf_time_unit`
(`frames` or `seconds`), and `kf_reference_dt_s`. `trials.json` records the trial
results and timing settings.
The final score is measured on the same split used to select those parameters.
It is a tuned score, not an independent accuracy estimate; calibration does
not guarantee improvement on other sequences.

Use `--tracker-config` to evaluate the saved configuration on a separate split
with ground truth and a compatible build. For example, after preparing a
validation build whose sequences are held out from calibration:

```bash
boxmot eval \
  --dataset mot17 \
  --split val \
  --build /srv/boxmot/materializations/HELD_OUT_BUILD \
  --tracker botsort \
  --tracker-config path/to/kf-tuning/best.yaml
```

`--tracker-config` also accepts a partial scalar YAML or a built-in preset such
as `botsort-mot17-ablation`. It overlays tracker defaults. Explicit runtime
flags such as `--asso-func` can override scalar parameters, but a calibrated
config's time units must match the selected mode. For example,
`--fixed-dt --tracker-config seconds-config.yaml` is rejected when that file
declares `kf_time_unit: seconds`. Recalibrate in the intended mode instead of
reinterpreting the saved values. Geometry,
backend, class selection, and per-class tracking remain separate CLI or dataset
controls, recorded in the trial report. Repeat `--per-class` if it was used
during fitting, and use the same geometry and backend.

### Live streams

The full five-parameter search uses ground truth to compare tracking results.
Run it on representative timestamped recordings, including dropped frames and
missed detections, then load the saved profile for live tracking. Keep the
same detector, geometry, timing mode, and reference interval.

BoostTrack and OccluBoost also support experimental online process-noise
adaptation with `adaptive_kf: true`. This learns from each track's prediction
errors; it does not learn measurement noise or initial uncertainty and does
not measure tracking accuracy. Incorrect associations can distort its estimates.
Enable it in the starting configuration during calibration if deployment will
use it, so the fitting and deployment behavior agree:

```yaml title="live-kf.yaml"
tracker: occluboost
variable_dt: true
adaptive_kf: true
```

Pass `--tracker occluboost --tracker-config live-kf.yaml --kf-tuning` to the
evaluation command, then use its `best.yaml` with
`boxmot track --tracker occluboost --source <stream> --tracker-config <best.yaml>`.
Other trackers support the calibrated static profile and interval-aware
prediction; they do not gain online noise adaptation from `--variable-dt`.

Live intervals must come from trustworthy source timestamps. A nominal-FPS
fallback cannot reveal unseen capture dropouts or reconnect duration; use
capture/PTS metadata or provide `Frame.timestamp_s` through the Python API.
Elapsed inference time is not a capture timestamp.

## Sequence parallelism

Evaluation replays each sequence as one isolated spawned-process job. Every
job constructs its own tracker from the immutable tracker spec, so tracker
state and native handles never cross sequence or process boundaries.
`--n-threads` sets the maximum number of sequence worker processes. The Rich
panel reports frame progress separately for every sequence while those jobs
run.

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
