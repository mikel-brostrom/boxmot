---
description: Precompute BoxMOT detections and embeddings for repeated evaluation and tuning runs.
---

# Generate Mode

`generate` builds reusable detection and embedding caches under `project/dets_n_embs/`. This is useful when you want `eval` and `tune` to reuse the same cached inputs across repeated experiments.

!!! example "Generate from CLI or Python"

    === "CLI"

        ```bash
        boxmot generate \
          --benchmark mot17-ablation
        ```

    === "Python"

        ```python
        from boxmot.configs import build_mode_namespace
        from boxmot.engine.evaluator import run_generate_dets_embs

        args = build_mode_namespace(
            "generate",
            {
                "data": "mot17-ablation",
                "detector": "yolo11s",
                "reid": "osnet_x0_25_msmt17",
                "project": "runs",
            },
            explicit_keys={"data", "detector", "reid", "project"},
        )

        run_generate_dets_embs(args)
        ```

## Notes

- `generate` accepts either `--benchmark` / `data=` or a direct dataset `--source`, but not both.
- The public high-level `BoxMOT` wrapper does not expose `generate()` yet, so the Python tab uses the lower-level evaluator entry point.
- Generated caches are later consumed automatically by `eval` and `tune`.

## CLI Reference

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: generate
    :style: table
    :prog_name: boxmot generate
