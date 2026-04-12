# Installation

## Requirements

BoxMOT supports Python `3.9` through `3.12`.

## Basic install

```bash
pip install boxmot
boxmot --help
```

This is enough for standard tracking, evaluation, tuning, and Python API usage as long as your detector and ReID dependencies are available in the environment.

## Research install

The `research` mode requires the optional research extra:

```bash
uv sync --extra research
```

That installs GEPA and the additional dependencies used by the code-evolution loop.

## Verify the install

!!! example "Verify"

    === "CLI"

        Check the CLI:

        ```bash
        boxmot --help
        boxmot track --help
        ```

    === "Python"

        Smoke-test the Python API:

        ```python
        from boxmot import Boxmot

        boxmot = Boxmot(detector="yolov8n", reid="osnet_x0_25_msmt17", tracker="bytetrack")
        print(boxmot)
        ```

## Next steps

- Use [Quickstart](../index.md) for a minimal path.
- Use [Choose a Mode](workflows.md) to decide between `track`, `generate`, `eval`, `tune`, `research`, and `export`.
