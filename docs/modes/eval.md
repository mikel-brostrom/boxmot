# Evaluate

Use `eval` to score tracking runs on MOT-style datasets with BoxMOT's in-repo MOT metrics.

## Examples

!!! example

    === "CLI"

        ```bash
        boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack --verbose
        ```

    === "Python"

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
        metrics = boxmot.val(experiment="mot17-ablation-yolox-lmbn")
        print(metrics)
        ```

Use a model-free dataset profile when the detector and ReID models should come
from the CLI options or runtime defaults instead of an experiment:

```bash
boxmot eval --dataset mot17 --split ablation --tracker boosttrack
```

`--dataset` and `--experiment` are mutually exclusive. An experiment selected
with `--experiment` remains the reproducible option when detector, ReID, and
detection-source choices must be fixed by configuration.

## Typical workflow

!!! example

    === "CLI"

        For repeated experiments:

        ```bash
        boxmot generate --experiment mot17-ablation-yolox-lmbn
        boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack
        ```

        This lets `eval` reuse precomputed detections and embeddings.

    === "Python"

        ```python
        from boxmot import BoxMOT

        boxmot = BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")
        metrics = boxmot.val(experiment="mot17-ablation-yolox-lmbn")
        print(metrics)
        ```

## Public detections

Select an experiment whose `detections.source` is `public`:

```bash
boxmot eval --experiment mot17-ablation-frcnn-lmbn --tracker boosttrack
boxmot eval --experiment mot17-ablation-sdp-lmbn --tracker boosttrack
boxmot eval --experiment mot17-ablation-dpm-lmbn --tracker boosttrack
```

The selected experiment identifies the public source in the central artifact
profile. The compatibility option `--detection-source` accepts `public` or
`private`, but a source-specific experiment ID is the reproducible way to
choose FRCNN, SDP, or DPM.

See [Experiment Workflows](../guides/experiments.md#detection-sources) for details on how public detections are resolved.

## Kalman filter noise tuning

Use `--tune-kf` to estimate per-sequence Kalman filter process and measurement noise (Q/R matrices) from the cached detections and ground truth before tracking:

```bash
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack --tune-kf
```

This is most useful for trackers with Kalman-filter-based motion models. It requires cached detections and ground truth to be available.

For runtime adaptation without ground truth, `boosttrack` and `occluboost`
expose the `adaptive_kf` tracker setting, which estimates noise online via the
Mehra (1970) method. It is a tracker configuration value, not a CLI flag. For
example, the Python facade can override it directly:

```python
from boxmot import BoxMOT

boxmot = BoxMOT(
    detector="yolov8n",
    reid="lmbn_n_duke",
    tracker="boosttrack",
    tracker_kwargs={"adaptive_kf": True},
)
metrics = boxmot.val(experiment="mot17-ablation-yolox-lmbn")
```

## Compare with TrackEval

Install the optional TrackEval reference implementation and request an independent comparison:

```bash
uv sync --extra yolo --extra trackeval
boxmot eval --experiment mot17-ablation-yolox-lmbn \
  --tracker boosttrack \
  --compare-trackeval
```

The report shows BoxMOT's in-repo metrics followed by `Δ vs TrackEval` rows. TrackEval reads the generated MOT files and runs its own MOTChallenge preprocessing, including distractor removal. This comparison currently supports AABB MOT15, MOT16, MOT17, and MOT20 benchmarks.

## Postprocessing

!!! example

    === "CLI"

        `eval` can apply optional postprocessing before scoring.
        Multiple steps can be chained with commas and are applied sequentially to the same result files:

        ```bash
        # Single step
        boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack --postprocessing gsi

        # Chained: GSI runs first, then GTA reads GSI's output
        boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker boosttrack --postprocessing gsi,gta
        ```

        Available steps:

        | Step | Description |
        | --- | --- |
        | `gsi` | Gaussian-smoothed interpolation — fills gaps and smooths trajectories |
        | `gbrc` | Gradient-boosting reconnection — ML-based interpolation and smoothing |
        | `gta` | Global tracklet association — offline split-and-connect across the full sequence |

    === "Python"

        `BoxMOT.val(...)` is the Python-facing validation entry point. Postprocessing details and metric interpretation are the same as in the CLI evaluation pipeline.

!!! warning "Chained steps overwrite in place"
    When chaining multiple postprocessing steps, each step reads the MOT result files, transforms them, and writes back to the same directory. The second step operates on the output of the first.

See [Evaluation and Postprocessing](../guides/evaluation.md).

See [Experiment Workflows](../guides/experiments.md) for cache reuse, MMOT experiment IDs, and replay image-loading behavior.

## Native C++ replay

Use `--tracker-backend cpp` to run the cached replay stage through a native tracker implementation:

```bash
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker bytetrack --tracker-backend cpp
boxmot eval --experiment mot17-ablation-yolox-lmbn --tracker ocsort --tracker-backend cpp
```

Native replay is currently available for `botsort`, `bytetrack`, `ocsort`,
`occluboost`, and `sfsort`. Select the implementation with the separate
`--tracker-backend` option; tracker names do not accept a `:cpp` suffix.

## Main outputs

- combined benchmark metrics such as `HOTA`, `MOTA`, and `IDF1`
- per-sequence summaries
- optional runtime timing summary with `--show-timing`
- MOT-style tracker outputs
- reused cache paths and evaluation artifacts in the run directory

See [Evaluation and Postprocessing](../guides/evaluation.md).

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: eval
    :style: table
    :prog_name: boxmot eval
