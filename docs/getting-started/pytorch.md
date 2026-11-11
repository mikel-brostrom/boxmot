# PyTorch and Source Setup

Use these instructions to select CPU-only or CUDA 13.0 PyTorch wheels, or to
run BoxMOT from source. The normal [installation](installation.md) uses PyPI's
default PyTorch build.

## Package installation

Install the repository's required `uv` version and create a virtual environment:

```bash
pip install "uv==0.12.4"
uv venv --python 3.11
```

Choose one backend:

```bash
# CPU-only PyTorch
uv pip install "boxmot[cpu]" "ultralytics>=8.4.58" --torch-backend=cpu

# Or CUDA 13.0 PyTorch, on a host with a compatible NVIDIA driver
uv pip install "boxmot[cu130]" "ultralytics>=8.4.58" --torch-backend=cu130
```

Activate `.venv` or run its executables directly, for example
`.venv/bin/boxmot --help` on Linux/macOS or `.venv\Scripts\boxmot.exe --help`
on Windows. Ultralytics supplies the detector used by the quickstart; omit it
when your application supplies detections.

The `--torch-backend` option selects the wheel index. Standard `pip` does not
apply the source mappings in `tool.uv.sources`, so adding `[cpu]` or `[cu130]`
to a plain `pip install` command does not select that PyTorch build.

## Source checkout

The repository defines mutually exclusive, lockfile-backed `cpu` and `cu130`
extras. Select exactly one, and add any workflow extras you need:

```bash
git clone https://github.com/mikel-brostrom/boxmot.git
cd boxmot
pip install "uv==0.12.4"
uv sync --extra cpu --extra yolo
uv run --no-sync python -m boxmot.engine.cli --help
```

Replace `--extra cpu` with `--extra cu130` for CUDA 13.0. The `yolo` extra
installs Ultralytics and YOLOX using the repository's YOLOX dependency and build
configuration. YOLOX builds a C++ extension, so a C++ compiler is required.

Repeat your selected extras on every later sync; `uv` does not remember them:

```bash
uv sync --extra cpu --extra yolo --extra evolve
uv run --no-sync boxmot tune --help
```

After syncing, use `uv run --no-sync` or `.venv` executables. A plain `uv run`
can replace the selected PyTorch build by syncing without your profile extra.
For examples shown as `boxmot ...` elsewhere in these docs, run
`uv run --no-sync boxmot ...` from your checkout. Run Python entry points as
modules from the repository root.

See [optional extras](installation.md#mode-specific-extras) for workflow
dependencies and [contributor setup](../contributing/index.md) for the full
development, testing, and documentation environment.
