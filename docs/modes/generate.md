# Generate

Use `generate` to precompute detections and embeddings that can be reused by later `eval`, `tune`, or `research` runs.

## Examples

!!! example

    === "CLI"

        Experiment-driven cache generation:

        ```bash
        boxmot generate --experiment mot17-ablation-yolox-lmbn
        ```

        Direct-source cache generation:

        ```bash
        boxmot generate \
          --source path/to/dataset \
          --detector yolov8n \
          --reid osnet_x0_25_msmt17
        ```

    === "Python"

        ```python
        from boxmot import BoxMOT

        experiment_cache = BoxMOT().generate(experiment="mot17-ablation-yolox-lmbn")
        print(experiment_cache.cache_dir)

        direct_cache = BoxMOT(
            detector="yolov8n",
            reid="osnet_x0_25_msmt17",
        ).generate(source="path/to/dataset")
        print(direct_cache.timings["frames"])
        ```

## Why generate first

Cache generation removes repeated detector and ReID work from later benchmark runs. That makes evaluation and tuning faster and more reproducible.

## What gets written

`generate` writes cached detector outputs and ReID embeddings below
`<project>/dets_n_embs/<dataset>/<split>/`. Detection producers and
ReID model/runtime identities add further subdirectories so compatible later
runs can reuse the artifacts without mixing incompatible caches. The run
`--name` is not part of this cache root.

## When to use it

- before repeated `eval` runs on the same experiment
- before `tune`, which evaluates many tracker parameter sets
- before `research`, which may evaluate many candidate code variants

## Public detections

Select a public-detection experiment instead of a model-backed experiment:

```bash
boxmot generate --experiment mot17-ablation-frcnn-lmbn
```

This resolves and downloads the public artifact declared in
`boxmot/configs/artifacts`, then generates ReID embeddings for it. Later
`eval` and `tune` runs with the same experiment and runtime overrides reuse the
cache.

MOT17 has source-specific FRCNN, SDP, and DPM experiment IDs. The compatibility
option `--detection-source` accepts only `public` or `private`; use an experiment
ID to identify the exact public producer.

See [Experiment Workflows](../guides/experiments.md) for cache reuse, MMOT experiment IDs, and replay image-loading behavior.

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: generate
    :style: table
    :prog_name: boxmot generate
