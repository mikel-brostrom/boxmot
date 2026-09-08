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
Sam2Mot requires full-frame detection-aligned masks and frames. Missing inputs
produce an actionable materialization error.

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

Use [`time-variant`](../modes/time-variant.md) to create a reproducible frame-loss
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
process with its own tracker instance. Use `--n-threads` to cap the active
sequence processes; the evaluation UI shows a separate frame-progress row for
each sequence. Use `--sequence NAME` to limit a diagnostic run to one sequence;
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
