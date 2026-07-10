# Benchmark Workflows

Use this guide when you want to run the benchmark-driven modes: `generate`, `eval`, `tune`, and `research`.

## Core idea

Benchmark workflows resolve the dataset, detector, and ReID defaults from self-contained YAMLs under `boxmot/configs/benchmarks/`. The first run generates detections and embeddings, and later runs reuse that cache.
Downloaded MOT-style benchmark datasets are cached under `boxmot/datasets/mot/`.
ReID datasets use the sibling cache root `boxmot/datasets/reid/`.

```bash
boxmot generate --benchmark mot17 --split ablation
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack
boxmot tune --benchmark mot17 --split ablation --tracker bytetrack
```

## Built-in benchmark ids

- `mot17` for MOT17 and the ablation split workflow
- `sportsmot` for SportsMOT
- `mmot` for the MMOT benchmark config backed by OBB `.npy` frames
- `mmot-mini` for a local MMOT-style OBB benchmark rooted at `assets/mmot-mini`

`mmot` is the CLI benchmark id for MMOT.

## Cache reuse

`generate`, `eval`, `tune`, and `research` share a cache key derived from the dataset, detector, and ReID configuration.

- Keep the same benchmark, split, detector, and ReID settings when you want later runs to reuse an existing cache.
- Changing any of those inputs creates a different cache bucket and forces regeneration.
- `--detection-source` also affects the cache key — public detection caches are stored separately from private detector caches.
- Native `--tracker-backend cpp` replay can still reuse the same detection cache, but trackers with native ReID write embeddings to a separate `__cpp` cache bucket.

## Public detections

Some benchmarks ship with public detections from the original challenge (e.g., MOT17 includes FRCNN, SDP, and DPM). Use `--detection-source` to generate and evaluate with these instead of the configured detector:

```bash
# Generate cache with public FRCNN detections
boxmot generate --benchmark mot17 --split ablation --detection-source frcnn

# Evaluate using the same public detections
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack --detection-source frcnn

# Tune against public detections
boxmot tune --benchmark mot17 --split ablation --tracker ocsort --detection-source frcnn --n-trials 10
```

Public detectors are defined in the benchmark YAML under `public_detectors`. The detection files are downloaded from the parquet repository and cached. ReID embeddings are generated automatically for the public detections.

`--detection-source public` uses the default public detector specified in the benchmark config's `download.public_detector` field.

## Replay image loading

Most cached replay runs do not need to read images at all. BoxMOT skips image loading when the selected tracker can work from cached detections and embeddings alone.

Trackers that need image data during replay, such as camera-motion-compensation paths, still read frames from the dataset during replay.

## Outputs

Benchmark workflows write reusable detection and embedding caches under the project run directory, plus tracker outputs and evaluation artifacts for the selected mode.

- `generate` writes the cache only.
- `eval` writes tracker outputs and MOT metric summaries.
- `tune` writes trial outputs and the best parameter set.
- `research` writes benchmark summaries for each evaluated code proposal.

## README benchmark table notes

The README benchmark table uses the following conventions and inputs:

- MOT17 ablation runs evaluate on the second half of the MOT17 training set because the public validation split is not available and the ablation detector was trained on the first half.
- MOT17 cells are shown as `Py (C++)`. The value in parentheses is the native replay path using `--tracker-backend cpp`.
- `—` means that tracker does not currently have a native replay backend for that benchmark path.
- MOT17 results use [pre-generated detections and embeddings](https://github.com/mikel-brostrom/boxmot/releases/download/v11.0.9/runs2.zip) with each tracker configured from its default repository settings.
- SportsMOT val results use the `yolox_x_sportsmot` detector and `lmbn_n_duke` ReID model.
- MMOT test metrics are class-averaged across all 8 categories, following the <a href="https://arxiv.org/abs/2510.12565">MMOT paper</a> convention, and use the `yolo11l_3ch` detector with the `lmbn_n_duke` ReID model.

## Related pages

- [Generate](../modes/generate.md)
- [Evaluate](../modes/eval.md)
- [Evaluation and Postprocessing](evaluation.md)
- [Benchmarks](../config/benchmarks.md)
