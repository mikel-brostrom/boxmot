# Benchmark Workflows

Use this guide when you want to run the benchmark-driven modes: `generate`, `eval`, `tune`, and `research`.

## Core idea

Benchmark workflows resolve the dataset, detector, and ReID defaults from the YAMLs under `boxmot/configs/datasets/`. The first run generates detections and embeddings, and later runs reuse that cache.

```bash
boxmot generate --benchmark mot17 --split ablation
boxmot eval --benchmark mot17 --split ablation --tracker boosttrack
boxmot tune --benchmark mot17 --split ablation --tracker bytetrack
```

## Built-in benchmark ids

- `mot17` for MOT17 and the ablation split workflow
- `sportsmot` for SportsMOT
- `mmot-obb` for the MMOT benchmark config backed by OBB `.npy` frames

`mmot-obb` is still the CLI benchmark id even when the surrounding docs and result tables refer to the benchmark as MMOT.

## Cache reuse

`generate`, `eval`, `tune`, and `research` share a cache key derived from the dataset, detector, and ReID configuration.

- Keep the same benchmark, split, detector, and ReID settings when you want later runs to reuse an existing cache.
- Changing any of those inputs creates a different cache bucket and forces regeneration.
- Native `--tracker-backend cpp` replay can still reuse the same detection cache, but trackers with native ReID write embeddings to a separate `__cpp` cache bucket.

## Replay image loading

Most cached replay runs do not need to read images at all. BoxMOT skips image loading when the selected tracker can work from cached detections and embeddings alone.

Trackers that need image data during replay, such as camera-motion-compensation paths, still read frames from the dataset during replay.

## Outputs

Benchmark workflows write reusable detection and embedding caches under the project run directory, plus tracker outputs and evaluation artifacts for the selected mode.

- `generate` writes the cache only.
- `eval` writes tracker outputs and TrackEval summaries.
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