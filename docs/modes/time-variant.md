# Time-variant dataset

Use `materialize --time-variant` to test tracking with irregular capture intervals. It selects
real frames from one MOT image sequence and reuses their cached detections and
embeddings. Images and annotations retain their original scene motion; frame
loss is simulated.

## Create a variant

Start with a complete [materialized build](materialize.md) for the source
dataset and split:

=== "CLI"

    ```bash
    boxmot materialize --time-variant \
      --dataset mot17 \
      --split ablation \
      --sequence MOT17-10-FRCNN \
      --build BUILD_ID \
      --seed 0
    ```

=== "Python"

    ```python
    from boxmot.engine.dataset_variants.workflow import main as create_time_variant

    result = create_time_variant(
        dataset="mot17",
        split="ablation",
        sequence="MOT17-10-FRCNN",
        build="BUILD_ID",
        seed=0,
    )
    ```

`--build` accepts a build ID or directory. Use `--build-root` and `--data-root`
when the cached build or source dataset lives outside the default roots.
The command reuses perception results from that build, so it does not run a
detector or appearance encoder. The source must be a MOT image sequence with
ground truth, AABB geometry, complete cached frame coverage, and at least 30 frames with
finite, strictly increasing capture timestamps. `--fps`, `--experiment`,
perception publication flags, executor options, and `--resume` cannot be
combined with `--time-variant`.

`--name` selects a new dataset identifier. The default is the lowercase sequence
name followed by `-variable-time`, such as `mot17-10-frcnn-variable-time`.
Existing output datasets are refused; choose a different name for another
variant. Source data and the parent build remain unchanged.

## Timing and annotations

The `bursty-v1` profile combines normal capture, sustained load, and frame loss.
For a 30 FPS source:

| Source interval | Selection pattern | Typical retained rate |
| --- | --- | --- |
| 18–37% of the sequence | Advance 2, 3, or 4 frames with probabilities 20%, 60%, 20% | 10 FPS |
| 69–83% of the sequence | Advance 1, 2, or 3 frames with probabilities 10%, 80%, 10% | 15 FPS |
| Remaining intervals | Advance 1 or 2 frames with probabilities 90%, 10% | About 27 FPS |

One blackout removes roughly 300 ms near 60% of the sequence, with its location
varied by up to 2% of the sequence length. The first and last frames are retained.
`--seed` makes frame selection and the blackout location reproducible. This is a
simulated delivery/drop pattern; it was not measured from a live network.

Retained frames keep their original capture timestamps. For a 30 FPS image
sequence, intervals are multiples of 1/30 second: removing frames increases
both elapsed time and the scene motion between observations. The command does
not invent timestamp jitter for consecutive images or interpolate new images.

Output frames receive consecutive frame numbers. Ground-truth rows are selected
and renumbered with the same mapping, preserving object identities, boxes,
classes, ignore flags, and visibility. Dropped frames are absent from the new
sequence. A timestamp sidecar preserves the irregular intervals independently
of frame numbering.

Inspect `variant.json` for the actual source and retained frame counts,
duration, effective FPS, interval histogram, outage bounds, and retained-to-source
frame mapping. These values depend on the selected split, source timeline, and
seed; they should accompany reported fixed-step versus variable-time results.

## Evaluate

The new dataset lives under `<data-root>/variants/<name>`; the default example
creates `datasets/mot/variants/mot17-10-frcnn-variable-time`. It contains:

- `dataset.yaml`, the reusable dataset configuration.
- `variant.json`, the sampling profile and source-frame provenance.
- `variable/MOT17-10-FRCNN/`, containing retained images, `timestamps.csv`,
  `seqinfo.ini`, and remapped `gt/gt.txt`.

The command prints the dataset YAML and derived build path, plus the retained
frame count and observed interval range. Pass the printed derived build as
`VARIANT_BUILD_ID` below:

```bash
boxmot eval \
  --dataset datasets/mot/variants/mot17-10-frcnn-variable-time/dataset.yaml \
  --split variable \
  --build VARIANT_BUILD_ID \
  --tracker bytetrack \
  --variable-dt
```

Run the same command with `--fixed-dt` to compare timing modes on identical
images, detections, and labels. Add `--calibrate-kf` to fit coordinate-wise noise
for the five covariance groups in the selected mode; see
[Kalman calibration](eval.md#kalman-calibration).

To view the tracking results after calibration, load the `calibrated.yaml` printed
by that run and select the same tracker. For the ByteTrack run above:

```bash
boxmot eval \
  --dataset datasets/mot/variants/mot17-10-frcnn-variable-time/dataset.yaml \
  --split variable \
  --build VARIANT_BUILD_ID \
  --tracker bytetrack \
  --tracker-config path/to/kf-tuning/calibrated.yaml \
  --variable-dt \
  --show --save
```

This replays the cached detections and embeddings with the calibrated tracker, using
the irregular capture timestamps. Preview and saved video hold each image
across its capture gap. Videos are saved under `<run>/videos/`; see
[view tracking results](eval.md#view-tracking-results) for playback details.
Alternatively, append `--show --save` to the calibration command to display and
record its tracker replay after calibration finishes.

The variant is one `variable` split, intended as a controlled stress test of
the source scene. It does not create an independent training or validation set.
Changing the random seed still reuses the same scene. Tune on development data
and evaluate on a separate sequence before claiming general improvement.

## Arguments

::: mkdocs-click
    :module: boxmot.engine.commands.materialize
    :command: materialize
    :prog_name: boxmot materialize
    :depth: 0
