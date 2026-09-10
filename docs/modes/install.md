# Install dependencies

Use `install` to add optional dependencies to the Python environment running
BoxMOT:

```bash
boxmot install --extra onnx
```

Download helpers, ReID backends, exporters, tuning, and research validate their
dependencies when used. Run this command explicitly when a dependency check
reports missing packages.

## Extras and requirements

Repeat `--extra` to prepare several workflows:

```bash
boxmot install --extra yolo --extra evolve --extra onnx
```

Extra definitions come from the installed BoxMOT distribution. Checks include
version constraints and active transitive dependencies, including nested extras
such as `ray[tune]`. Installation is skipped when all selected requirements are
satisfied. Otherwise, the package manager receives the full requested constraint
set, including bounds that are already satisfied, to resolve the dependency
environment. The command targets dependency packages only.

Use `--requirement` for a package requirement, including a version constraint or
package extra. It can be repeated or combined with `--extra`:

```bash
boxmot install --requirement 'onnxruntime==1.24.3'
```

Requirements must use indexed package names, extras, or version constraints.
Install packages from direct URLs with `pip` or `uv`.

See [mode-specific extras](../getting-started/installation.md#mode-specific-extras)
for the available workflow selections. For a source checkout managed with
`uv sync`, continue using the [PyTorch profile workflow](../getting-started/installation.md#select-a-pytorch-profile)
to preserve the selected CPU or CUDA build. The `cpu` and `cu130` profile extras
are not accepted by `boxmot install`.

## Select the Python environment

Invoke the CLI as a module with the interpreter that will run your application:

```bash
python -m boxmot.engine.cli install --extra onnx

# Existing repository virtual environment
uv run --no-sync python -m boxmot.engine.cli install --extra onnx
```

Installation targets that interpreter, including when `uv` performs the
installation. Satisfied requirements need no installation.

## TensorRT

TensorRT export uses ONNX as an intermediate. Install its Python dependencies
explicitly, including NVIDIA's package index:

```bash
python -m boxmot.engine.cli install \
  --extra onnx \
  --requirement 'nvidia-tensorrt' \
  --extra-index-url https://pypi.ngc.nvidia.com
```

The TensorRT runtime also needs compatible Python, CUDA, and NVIDIA driver
versions. See [TensorRT troubleshooting](../guides/troubleshooting.md#tensorrt-is-installed-but-import-still-fails)
if the installed package cannot be imported.

## CLI arguments

::: mkdocs-click
    :module: boxmot.engine.commands.install
    :command: install
    :depth: 0
    :style: table
    :prog_name: boxmot install
