# Installation

## Requirements

BoxMOT supports Python `3.10` through `3.13`.

## Basic install

```bash
pip install boxmot
boxmot --help
```

This installs the core package. It is enough for simple tracking and Python API usage when your detector and ReID backends are already available in the environment.

## Mode-specific extras

BoxMOT keeps heavier workflow dependencies optional. Install the extras that match the modes and export targets you plan to use.

| Workflow | PyPI install | Source checkout with `uv` | Notes |
| --- | --- | --- | --- |
| `track`, `generate`, `eval` with common YOLO backends | `pip install "boxmot[yolo]"` | `uv sync --extra yolo` | Preinstalls Ultralytics and YOLOX. If you skip this, BoxMOT can install some detector packages on first use. |
| `tune` | `pip install "boxmot[evolve]"` | `uv sync --extra evolve` | Installs Ray Tune, Optuna, Plotly, and related tuning dependencies. |
| `research` | `pip install "boxmot[research]"` | `uv sync --extra research` | Installs GEPA for the code-evolution loop. |
| `export --include onnx` | `pip install "boxmot[onnx]"` | `uv sync --extra onnx` | The default export path uses ONNX. |
| `export --include openvino` | `pip install "boxmot[openvino]"` | `uv sync --extra openvino` | Usually paired with `--include onnx`. |
| `export --include tflite` | `pip install "boxmot[tflite]"` | `uv sync --extra tflite` | Python `3.12` on Linux or Windows only. |

You can combine extras when needed:

```bash
uv sync --extra yolo --extra evolve --extra research
pip install "boxmot[yolo,evolve,research]"
```

`boxmot export --include engine` requires NVIDIA TensorRT to be available in the environment. It is not provided as a BoxMOT extra.

## Native C++ backends

Native C++ tracker backends are built lazily the first time you select `--tracker-backend cpp`. They are currently available for `botsort`, `bytetrack`, `ocsort`, `occluboost`, and `sfsort`.

Install the native build tools before using them:

- C++17 compiler
- CMake 3.16+
- OpenCV 4.x
- Eigen3 3.3+

Example:

```bash
boxmot track --detector yolov8n --tracker bytetrack --tracker-backend cpp --source video.mp4
boxmot eval --benchmark mot17 --split ablation --tracker bytetrack --tracker-backend cpp
```

The generated build files are kept under `build/native/<tracker>/`.

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
- Use [Modes Overview](../modes/index.md) to decide between `track`, `generate`, `eval`, `tune`, `research`, and `export`.
- Use the workflow table above to add the extras your workflow needs.
