# Generate

Use `generate` to precompute detections and embeddings that can be reused by later `eval`, `tune`, or `research` runs.

## Examples

!!! example

    === "CLI"

        Benchmark-driven cache generation:

        ```bash
        boxmot generate --benchmark mot17 --split ablation
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

        benchmark_cache = BoxMOT().generate(benchmark="mot17", split="ablation")
        print(benchmark_cache.cache_dir)

        direct_cache = BoxMOT(
            detector="yolov8n",
            reid="osnet_x0_25_msmt17",
        ).generate(source="path/to/dataset")
        print(direct_cache.timings["frames"])
        ```

## Why generate first

Cache generation removes repeated detector and ReID work from later benchmark runs. That makes evaluation and tuning faster and more reproducible.

## What gets written

`generate` writes cached detector outputs and ReID embeddings under the configured project/name directory so later runs can reuse them.

## When to use it

- before repeated `eval` runs on the same benchmark
- before `tune`, which evaluates many tracker parameter sets
- before `research`, which may evaluate many candidate code variants

## Public detections

Use `--detection-source` to cache public MOTChallenge detections instead of running a detector:

```bash
boxmot generate --benchmark mot17 --split ablation --detection-source frcnn
```

This downloads the public detection files from the benchmark config and generates ReID embeddings for them. Later `eval` and `tune` runs with the same `--detection-source` reuse this cache.

Available sources for MOT17: `frcnn`, `sdp`, `dpm`, or `public` (uses the default defined in the benchmark YAML).

See [Benchmark Workflows](../guides/benchmarks.md) for cache reuse, MMOT benchmark ids, and replay image-loading behavior.

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: generate
    :style: table
    :prog_name: boxmot generate
