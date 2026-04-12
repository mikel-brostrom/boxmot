# Generate

Use `generate` to precompute detections and embeddings that can be reused by later `eval`, `tune`, or `research` runs.

## Examples

!!! example

    === "CLI"

        Benchmark-driven cache generation:

        ```bash
        boxmot generate --benchmark mot17-ablation
        ```

        Direct-source cache generation:

        ```bash
        boxmot generate \
          --source path/to/dataset \
          --detector yolov8n \
          --reid osnet_x0_25_msmt17
        ```

    === "Python"

        BoxMOT does not expose a first-class public `generate(...)` workflow on the high-level Python facade.

        Use the CLI for cache generation, then use Python APIs such as `Boxmot.val(...)`, `Boxmot.tune(...)`, or `Boxmot.track(...)` to consume benchmark configs and tracking results.

## Why generate first

Cache generation removes repeated detector and ReID work from later benchmark runs. That makes evaluation and tuning faster and more reproducible.

## What gets written

`generate` writes cached detector outputs and ReID embeddings under the configured project/name directory so later runs can reuse them.

## When to use it

- before repeated `eval` runs on the same benchmark
- before `tune`, which evaluates many tracker parameter sets
- before `research`, which may evaluate many candidate code variants

## CLI Arguments

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: generate
    :style: table
    :prog_name: boxmot generate
