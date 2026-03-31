---
description: Install BoxMOT and run your first tracking, benchmark, and export commands.
---

# Quickstart

BoxMOT supports Python `3.9` through `3.12`. For local development and reproducible contributor workflows, use Python `3.11` with `uv`.

## Installation

!!! example "Install BoxMOT"

    === "PyPI"

        Install the published package when you want the CLI and Python tracker API quickly.

        ```bash
        pip install boxmot
        boxmot --help
        ```

        This is the lightest setup, but some detector backends, tuning features, or export targets may require additional optional dependencies.

    === "Source with uv"

        Clone the repository when you want the full BoxMOT workflow, including docs, tests, detector extras, tuning, and export backends.

        ```bash
        git clone https://github.com/mikel-brostrom/boxmot.git
        cd boxmot

        pip install uv
        uv sync --all-extras --all-groups

        uv run python -m boxmot.engine.cli --help
        ```

        Use `uv run ...` from the repository root so BoxMOT executes with the correct package context.

    === "Docker"

        BoxMOT ships with a CUDA-enabled `Dockerfile` for GPU environments.

        ```bash
        docker build -t boxmot .
        docker run -it --gpus all boxmot bash
        ```

        Inside the container, BoxMOT is installed from source with `uv sync --all-extras --all-groups`.

## First Commands

The BoxMOT CLI is organized around modes:

```text
track      run detector + tracker on live or file-based inputs
generate   precompute detections and embeddings for later reuse
eval       benchmark on MOT-style datasets
tune       search tracker hyperparameters from tracker YAML files
export     export ReID backbones to deployment formats
```

Start by checking the root help and one mode-specific page:

!!! example "First execution"

    === "CLI"

        ```bash
        boxmot --help
        boxmot track --help
        ```

    === "Python"

        ```python
        from boxmot import boxmot

        tracker = boxmot(tracker="bytetrack")

        print(type(tracker).__name__)  # BoxMOT
        ```

## First Tracking Run

Run a detector, ReID model, and tracker together on a local video:

!!! example "Track from CLI or Python"

    === "CLI"

        ```bash
        boxmot track \
          --detector yolov8n \
          --reid osnet_x0_25_msmt17 \
          --tracker botsort \
          --source video.mp4 \
          --save
        ```

    === "Python"

        ```python
        from boxmot import boxmot

        tracker = boxmot(
            detector="yolov8n",
            reid="osnet_x0_25_msmt17",
            tracker="botsort",
        )

        results = tracker.track(source="video.mp4", save=True)
        print(results.video_path)
        ```

Use `--show` for live display, `--show-trajectories` for path overlays, and `--save-txt` when you want a MOT-style text output alongside the annotated video. If you already have your own detections or need a custom frame loop, open [Python API](usage/python.md) for direct tracker usage.

## First Benchmark Run

Benchmark workflows are driven by benchmark YAML bundles under `boxmot/configs/benchmarks/`.

```bash
boxmot eval --benchmark mot17-mini --tracker bytetrack
```

That single command resolves:

- the dataset config under `boxmot/configs/datasets/`
- the detector profile under `boxmot/configs/detectors/`
- the ReID profile under `boxmot/configs/reid/`
- the tracker defaults under `boxmot/configs/trackers/bytetrack.yaml`

If cached detections and embeddings do not exist yet, `eval` generates them first and then reuses them on later runs.

## Generate Reusable Caches

If you know you will evaluate or tune repeatedly, generate the detector and embedding cache first:

```bash
boxmot generate --benchmark mot17-ablation
```

For an already-local dataset without a benchmark config, use a direct source path instead:

```bash
boxmot generate \
  --source ./assets/MOT17-mini/train \
  --detector yolov8n \
  --reid osnet_x0_25_msmt17
```

## Export a ReID Model

BoxMOT exports ReID backbones, not detector weights:

```bash
boxmot export --weights osnet_x0_25_msmt17.pt --include onnx --include engine --dynamic
```

## Common Gotchas

- Use `--detector`, `--reid`, and `--tracker`. Legacy aliases such as `--yolo-model` are not supported.
- `track` works from `--source`. `eval` and `tune` require `--benchmark`.
- `generate` accepts either `--benchmark` or `--source`, but not both.
- OBB tracking requires both OBB detections and an OBB-capable tracker.

## Next Steps

- Open [CLI Usage](usage/cli.md) for the full command model.
- Open [Track Mode](modes/track.md) if you want live or file-based inference.
- Open [Evaluate Mode](modes/eval.md) if you want TrackEval metrics.
- Open [Python API](usage/python.md) for both the high-level workflow wrapper and low-level tracker building blocks.
